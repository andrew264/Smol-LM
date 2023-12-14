import re
from typing import Optional

import torch
import torch.nn as nn
from flash_attn.ops.fused_dense import FusedDense
from flash_attn.ops.rms_norm import RMSNorm
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from model import ModelConfig, Tokenizer
from model.block import TransformerBlock


def precompute_freqs_cis(
        seq_len: int, n_elem: int, base: float = 10000.0
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


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


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.causal_mask = None
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = FusedDense(config.hidden_size, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = config.max_batch_size
        self.max_seq_length = config.max_position_embeddings
        self.gradient_checkpointing = config.gradient_checkpointing

        self.freqs_cis = precompute_freqs_cis(self.config.max_position_embeddings,
                                              self.config.hidden_size // self.config.num_attention_heads,
                                              self.config.rope_theta).cuda()
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).cuda()

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

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if input_pos is None:
            device = idx.device
            input_pos = torch.arange(0, idx.shape[-1], dtype=torch.long, device=device)
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer.__call__, x, input_pos, freqs_cis, mask, use_reentrant=True)
            else:
                x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, tokenizer: Tokenizer, max_tokens: int, stream: bool = True,
                 **sampling_kwargs) -> str:
        return_output = ''
        while len(prompt) < max_tokens:
            logits = self(prompt.view(1, -1))
            idx_next, _ = sample(logits, **sampling_kwargs)
            idx = idx_next.tolist()[0]
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
