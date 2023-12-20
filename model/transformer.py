import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.ops.fused_dense import FusedDense
from flash_attn.ops.rms_norm import RMSNorm
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from model import ModelConfig, Tokenizer
from model.block import TransformerBlock


def load_balancing_loss_func(gate_logits: Tensor | Tuple, num_experts: int = None, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    # treat `top_k` as tokens (shape is `top_k X [batch_size X sequence_length]`)
    selected_experts = selected_experts.reshape(-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    expert_mask = torch.max(expert_mask, dim=-2).values

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-1))
    return overall_loss * num_experts


def multinomial_sample_one_no_sync(probs_sort):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.causal_mask = None
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = FusedDense(config.hidden_size, config.vocab_size, bias=False)

        self.max_batch_size = config.max_batch_size
        self.max_seq_length = config.max_position_embeddings
        self.gradient_checkpointing = config.gradient_checkpointing
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts

        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).cuda()
        self.freqs_cis = precompute_freqs_cis(self.config.hidden_size // self.config.num_attention_heads,
                                              self.config.max_position_embeddings * 2,
                                              self.config.rope_theta).cuda()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, x: Tensor, y: Optional[Tensor] = None, start_pos: int = 0) -> tuple[Tensor, Optional[Tensor]]:
        seq_length = x.shape[1]
        mask = self.causal_mask[:seq_length, :seq_length]
        x = self.tok_embeddings(x)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seq_length]

        all_router_logits = ()
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x, router_logits = checkpoint(layer.__call__, x, start_pos, mask, freqs_cis, use_reentrant=False)
            else:
                x, router_logits = layer(x, start_pos, mask, freqs_cis=freqs_cis)
            all_router_logits += (router_logits,)
        x = self.norm(x)
        logits = self.output(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            aux_loss = load_balancing_loss_func(all_router_logits, self.num_experts, top_k=2)
            loss += self.router_aux_loss_coef * aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: Tensor, tokenizer: Tokenizer, max_tokens: int, stream: bool = True,
                 **sampling_kwargs) -> str:
        return_output = ''
        prev_pos = 0
        tokens = torch.full((1, max_tokens), 0, dtype=torch.long, device="cuda")
        tokens[:, :prompt.shape[-1]] = prompt
        for cur_pos in range(prompt.shape[-1], max_tokens):
            logits, _ = self(tokens[:, prev_pos: cur_pos], start_pos=prev_pos)
            idx_next, _ = sample(logits, **sampling_kwargs)
            idx = idx_next.tolist()[0]
            tokens[:, cur_pos] = idx
            prev_pos = cur_pos
            if idx == tokenizer.eos_id or idx == tokenizer.pad_id:
                break
            out = tokenizer.decode_piece(idx)
            out = out.replace('▁', ' ')
            if match := re.match(r'<0x([0-9a-fA-F]+)>', out):
                out = bytes.fromhex(match.group(1)).decode('utf-8', errors='ignore')
            return_output += out
            if stream:
                print(out, end='', flush=True)
            prompt = torch.cat([prompt, idx_next], dim=-1)
        if stream:
            print()
        return return_output
