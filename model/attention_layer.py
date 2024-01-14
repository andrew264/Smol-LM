from typing import Optional

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.ops.fused_dense import FusedDense
from torch import Tensor
from torch.nn import functional as F

from model import ModelConfig


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = FusedDense(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = FusedDense(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = FusedDense(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = FusedDense(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.k_cache = None
        self.v_cache = None

    def forward(self, x: Tensor, mask: Optional[Tensor], freqs_cis: Tensor) -> Tensor:
        bsz, seqlen, _ = x.size()

        query_states: Tensor = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        key_states: Tensor = self.k_proj(x).view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
        value_states: Tensor = self.v_proj(x).view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

        query_states, key_states = apply_rotary_emb(query_states, key_states, freqs_cis)

        if query_states.device.type == "cuda" and mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        kv_seq_len = key_states.shape[1]

        if self.training:
            use_sliding_windows = 0 < self.config.sliding_window < kv_seq_len
            attn_output = self._flash_attn_forward(query_states, key_states, value_states, mask,
                                                   use_sliding_windows=use_sliding_windows, )
        else:
            # key value caching
            if kv_seq_len > 1:  # reset cache when kv_seq_len > 1
                self.k_cache, self.v_cache = None, None
            if self.k_cache is None or self.v_cache is None:
                cache_shape = [bsz, 0, self.num_key_value_heads, self.head_dim]
                self.k_cache = torch.zeros(cache_shape, dtype=x.dtype, device=x.device, )
                self.v_cache = torch.zeros(cache_shape, dtype=x.dtype, device=x.device, )
            self.k_cache = torch.cat([self.k_cache, key_states], dim=1)
            self.v_cache = torch.cat([self.v_cache, value_states], dim=1)
            use_sliding_windows = 0 < self.config.sliding_window < self.k_cache.shape[1]

            attn_output = self._flash_attn_forward(query_states, self.k_cache, self.v_cache, mask,
                                                   use_sliding_windows=use_sliding_windows, )

        attn_output = attn_output.contiguous().view(bsz, seqlen, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output

    def _flash_attn_forward(self, query_states, key_states, value_states, attention_mask, dropout=0.0,
                            use_sliding_windows=False, ) -> Tensor:
        causal = True
        sliding_window = (self.config.sliding_window, self.config.sliding_window) if use_sliding_windows else (-1, -1)
        if attention_mask is not None:
            bsz, q_len, _, _ = query_states.size()
            qkv = torch.stack([query_states, key_states, value_states], dim=2)
            qkv = qkv.reshape(bsz, q_len, -1)
            qkv, indices, cu_q_lens, max_s = unpad_input(qkv, attention_mask)
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
            attn_output_unpad = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_q_lens,
                max_s,
                dropout_p=dropout,
                causal=True,
                window_size=sliding_window
            )
            attn_output_unpad = attn_output_unpad.reshape(-1, self.num_heads * self.head_dim)
            attn_output = pad_input(attn_output_unpad, indices, bsz, q_len)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                causal=causal,
                window_size=sliding_window
            )

        return attn_output
