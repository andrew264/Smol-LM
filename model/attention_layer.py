from typing import Optional

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.ops.fused_dense import FusedDense
from torch import Tensor
from torch.nn import functional as F

from model import ModelConfig


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        if (self.num_heads % self.num_key_value_heads) != 0:
            raise ValueError(
                f"num_key_value_heads must divide evenly into num_heads (got `num_key_value_heads`: "
                f"{self.num_key_value_heads} and `num_heads`: {self.num_heads})."
            )
        total_head_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        self.qkv_proj = FusedDense(self.hidden_size, total_head_dim, bias=False)
        self.o_proj = FusedDense(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, interleaved=True)

        self.k_cache = None
        self.v_cache = None

    def update_kv_cache(self, key_states: Tensor, value_states: Tensor, input_pos: int) -> tuple[Tensor, Tensor]:
        if self.k_cache is None or self.v_cache is None:
            bsz = key_states.shape[0]
            device = key_states.device
            dtype = key_states.dtype
            cache_shape = [bsz, self.max_position_embeddings, self.num_key_value_heads, self.head_dim]
            self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device, )
            self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device, )
        seqlen = key_states.shape[1]
        total_len = input_pos + seqlen
        self.k_cache[:, input_pos:total_len] = key_states
        self.v_cache[:, input_pos:total_len] = value_states
        return self.k_cache[:, :total_len], self.v_cache[:, :total_len]

    def forward(self, x: Tensor, mask: Optional[Tensor], input_pos: Optional[int] = None) -> Tensor:
        bsz, seqlen, _ = x.size()

        if self.num_heads != self.num_key_value_heads:
            kv_size = self.num_key_value_heads * self.head_dim
            query_states, kv_states = self.qkv_proj(x).split([self.hidden_size, 2 * kv_size], dim=-1)
            query_states = query_states.view(bsz, seqlen, self.num_heads, self.head_dim)
            kv_states: Tensor = kv_states.view(bsz, seqlen, 2, self.num_key_value_heads, self.head_dim)

            query_states, kv_states = self.rotary_emb(query_states, kv_states, seqlen_offset=input_pos or 0)
            key_states, value_states = kv_states.unbind(dim=2)
        else:
            qkv_states = self.qkv_proj(x).view(bsz, seqlen, 3, self.num_heads, self.head_dim)
            qkv_states = self.rotary_emb(qkv_states, seqlen_offset=input_pos or 0)
            query_states, key_states, value_states = qkv_states.unbind(dim=2)

        if not self.training and input_pos is not None:
            key_states, value_states = self.update_kv_cache(key_states, value_states, input_pos=input_pos)

        if query_states.device.type == "cuda" and mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        use_sliding_windows = 0 < self.config.sliding_window < key_states.shape[1]
        attn_output = self._flash_attn_forward(query_states, key_states, value_states, mask,
                                               use_sliding_windows=use_sliding_windows, )

        attn_output = attn_output.contiguous().view(bsz, seqlen, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output

    def _flash_attn_forward(self, query_states, key_states, value_states, attention_mask, dropout=0.0,
                            use_sliding_windows=False, ) -> Tensor:
        causal = True
        sliding_window = (self.config.sliding_window, self.config.sliding_window) if use_sliding_windows else (-1, -1)
        if attention_mask is not None:
            batch_size, query_length = query_states.shape[:2]
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
                causal=causal,
                window_size=sliding_window
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
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
