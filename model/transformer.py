from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from model import ModelConfig
from model.block import TransformerBlock
from model.norm import RMSNorm
from model.utils import find_multiple


def precompute_freqs_cis(
        seq_len: int, n_elem: int, base: float = 10000.0
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16, device='cpu'):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))

    @torch.no_grad()
    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2], f"{input_pos.shape[0]} != {k_val.shape[2]}"

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.causal_mask = None
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.gradient_checkpointing = config.gradient_checkpointing

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

    def setup_caches(self, max_batch_size, max_seq_length, device='cuda'):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.register_module('kv_cache',
                                        KVCache(max_batch_size, max_seq_length, self.config.num_key_value_heads,
                                                head_dim, device=device))

        self.freqs_cis = precompute_freqs_cis(self.config.max_position_embeddings,
                                              self.config.hidden_size // self.config.num_attention_heads,
                                              self.config.rope_theta).to(device)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).to(device)

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
