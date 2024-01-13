from typing import Optional

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
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


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch,


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
        use_sliding_windows = 0 < self.config.sliding_window < kv_seq_len

        if self.training:
            attn_output = self._flash_attn_forward(query_states, key_states, value_states, mask, seqlen,
                                                   use_sliding_windows=use_sliding_windows, )
        else:
            # key value caching
            if self.k_cache is None or self.v_cache is None:
                cache_shape = [bsz, 0, self.num_key_value_heads, self.head_dim]
                self.k_cache = torch.zeros(cache_shape, dtype=x.dtype, device=x.device, )
                self.v_cache = torch.zeros(cache_shape, dtype=x.dtype, device=x.device, )
            self.k_cache = torch.cat([self.k_cache, key_states], dim=1)
            self.v_cache = torch.cat([self.v_cache, value_states], dim=1)

            attn_output = self._flash_attn_forward(query_states, self.k_cache, self.v_cache, mask, seqlen,
                                                   use_sliding_windows=use_sliding_windows, )

        attn_output = attn_output.contiguous().view(bsz, seqlen, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output

    def _flash_attn_forward(self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0,
                            softmax_scale=None, use_sliding_windows=False, ) -> Tensor:
        causal = True
        sliding_window = (self.config.sliding_window, self.config.sliding_window) if use_sliding_windows else (-1, -1)
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=sliding_window
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=sliding_window
            )

        return attn_output

    @staticmethod
    def _upad_input(query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len:]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
