from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from flash_attn.ops.fused_dense import FusedDense
from flash_attn.ops.rms_norm import RMSNorm
from tokenizers import Tokenizer
from torch import Tensor
from transformers import LogitsProcessorList

from model import ModelConfig
from model.block import TransformerBlock


def load_balancing_loss_func(gate_logits: Tensor | Tuple, num_experts: int = None, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
    """
    if not any(gate_logits):  # if gate_logits is None or empty tuple
        return 0

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


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
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

        self.loss_fn = CrossEntropyLoss(inplace_backward=True)

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

    def forward(self, x: Tensor, y: Optional[Tensor] = None, input_pos: Optional[int] = None,
                mask: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        x = self.tok_embeddings(x)

        all_router_logits = ()
        for i, layer in enumerate(self.layers):
            x, router_logits = layer(x, mask, input_pos=input_pos)
            all_router_logits += (router_logits,)
        x = self.norm(x)
        logits = self.output(x)
        loss = None
        if y is not None:
            batch_size, seq_length, vocab_size = logits.shape
            loss = self.loss_fn(logits.view(batch_size * seq_length, vocab_size),
                                y.view(batch_size * seq_length),
                                )
            if self.config.is_moe:
                aux_loss = load_balancing_loss_func(all_router_logits, self.num_experts, top_k=2)
                loss += self.router_aux_loss_coef * aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: Tensor, max_tokens: int, logits_processors: Optional[LogitsProcessorList] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 stream: bool = True) -> list[int]:
        return_output = []
        prev_pos = 0
        pad_id, eos_id = 0, 1
        tokens = prompt.unsqueeze(0)
        for cur_pos in range(prompt.shape[-1], max_tokens):
            logits, _ = self(tokens[:, prev_pos: cur_pos], input_pos=prev_pos)
            logits = logits[:, -1]
            # logits = F.log_softmax(logits, dim=-1)

            if logits_processors is not None:
                logits = logits_processors(input_ids=tokens, scores=logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            idx = next_token.item()
            tokens = torch.cat([tokens, next_token], dim=-1)
            prev_pos = cur_pos
            if idx in [pad_id, eos_id]:
                break
            return_output += [idx]
            if stream:
                if tokenizer is None:
                    raise ValueError("Tokenizer must be provided if stream is True.")
                print(tokenizer.decode([idx]), end='', flush=True)
            prompt = torch.cat([prompt, next_token.squeeze(1)], dim=-1)
        if stream:
            print()
        return return_output
