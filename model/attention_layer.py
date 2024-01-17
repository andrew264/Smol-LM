from typing import Optional

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_qkvpacked_func, \
    flash_attn_kvpacked_func
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
        self.max_position_embeddings = config.max_position_embeddings
        use_sliding_windows = 0 < self.config.sliding_window < self.max_position_embeddings
        self.sliding_window = (config.sliding_window, config.sliding_window) if use_sliding_windows else (-1, -1)

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

        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, )

        self.kv_cache = None

    def update_kv_cache(self, kv_states: Tensor, input_pos: int) -> Tensor:
        if self.kv_cache is None:
            bsz = kv_states.shape[0]
            device = kv_states.device
            dtype = kv_states.dtype
            cache_shape = [bsz, self.max_position_embeddings, 2, self.num_key_value_heads, self.head_dim]
            self.kv_cache = torch.zeros(cache_shape, dtype=dtype, device=device, )
        total_len = input_pos + kv_states.shape[1]
        self.kv_cache[:, input_pos:total_len] = kv_states
        return self.kv_cache[:, :total_len]

    def forward(self, x: Tensor, mask: Optional[Tensor], input_pos: Optional[int] = None) -> Tensor:
        bsz, seqlen, _ = x.size()

        if self.num_heads == self.num_key_value_heads:  # self-attention
            qkv_states = self.qkv_proj(x).view(bsz, seqlen, 3, self.num_heads, self.head_dim)
            qkv_states = self.rotary_emb(qkv_states, seqlen_offset=input_pos or 0)
            if self.training:
                attn_output = (
                    flash_attn_qkvpacked_func(qkv_states, causal=True,
                                              window_size=self.sliding_window)  # training - no mask
                    if mask is None
                    else self._flash_attn_forward(*qkv_states.unbind(dim=2), mask)  # training - mask
                )
            else:  # inference in self-attention
                if input_pos is not None:
                    kv_states = self.update_kv_cache(qkv_states[:, :, 1:], input_pos)
                    attn_output = flash_attn_kvpacked_func(qkv_states[:, :, 0], kv_states,
                                                           causal=True,
                                                           window_size=self.sliding_window)  # inference - with cache
                else:
                    attn_output = flash_attn_qkvpacked_func(qkv_states, causal=True,
                                                            window_size=self.sliding_window)  # inference - no cache
        else:  # MQ/GQ Attention
            kv_size = self.num_key_value_heads * self.head_dim
            q_state, kv_states = self.qkv_proj(x).split([self.hidden_size, 2 * kv_size], dim=-1)
            q_state = q_state.view(bsz, seqlen, self.num_heads, self.head_dim)
            kv_states: Tensor = kv_states.view(bsz, seqlen, 2, self.num_key_value_heads, self.head_dim)
            q_state, kv_states = self.rotary_emb(q_state, kv_states, seqlen_offset=input_pos or 0)
            if self.training:
                attn_output = (
                    flash_attn_kvpacked_func(q_state, kv_states, causal=True,
                                             window_size=self.sliding_window)  # training - no mask
                    if mask is None
                    else self._flash_attn_forward(q_state, *kv_states.unbind(dim=2), mask)  # training - mask
                )
            else:  # inference in MQ/GQ Attention
                if input_pos is not None:
                    kv_states = self.update_kv_cache(kv_states, input_pos)
                    use_sliding_windows = 0 < self.config.sliding_window < kv_states.shape[1]
                    attn_output = flash_attn_kvpacked_func(q_state, kv_states, causal=True,
                                                           window_size=self.sliding_window)  # inference - with cache
                else:
                    attn_output = flash_attn_kvpacked_func(q_state, kv_states, causal=True,
                                                           window_size=self.sliding_window)  # inference - no cache

        attn_output = attn_output.contiguous().view(bsz, seqlen, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _flash_attn_forward(self, query_states: Tensor, key_states: Tensor, value_states: Tensor,
                            attention_mask: Tensor, dropout=0.0, ) -> Tensor:
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
                causal=True,
                window_size=self.sliding_window
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # should not reach here but just in case
            print("WARNING: no attention mask provided, something is wrong.")
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                causal=True,
                window_size=self.sliding_window
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
