from typing import Optional

import torch
import torch.nn as nn
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.ops.fused_dense import FusedDense
from torch import Tensor
from torch.nn import functional as F

from model import ModelConfig


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

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
        self.qkv_proj = FusedDense(self.hidden_size, total_head_dim, bias=config.attention_bias)
        self.o_proj = FusedDense(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

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

    def forward(self, x: Tensor, attention_mask: Optional[Tensor], input_pos: Optional[int] = None) -> Tensor:
        bsz, seqlen, _ = x.size()
        is_causal = attention_mask is None and seqlen > 1
        qkv_states = self.qkv_proj(x)

        if self.num_heads == self.num_key_value_heads:  # self-attention
            qkv_states = qkv_states.view(bsz, seqlen, 3, self.num_heads, self.head_dim)
            qkv_states = self.rotary_emb(qkv_states, seqlen_offset=input_pos or 0)
            if input_pos is None:
                attn_output = self._sdpa(*qkv_states.unbind(dim=2), attention_mask=attention_mask, is_causal=is_causal)
            else:  # inference in self-attention
                kv_states = self.update_kv_cache(qkv_states[:, :, 1:], input_pos)
                attn_output = self._sdpa(qkv_states[:, :, 0], *kv_states.unbind(dim=2),
                                         attention_mask=attention_mask, is_causal=is_causal)
        else:  # MQ/GQ Attention
            kv_size = self.num_key_value_heads * self.head_dim
            q_state, kv_states = qkv_states.split([self.hidden_size, 2 * kv_size], dim=-1)
            q_state = q_state.view(bsz, seqlen, self.num_heads, self.head_dim)
            kv_states: Tensor = kv_states.view(bsz, seqlen, 2, self.num_key_value_heads, self.head_dim)
            q_state, kv_states = self.rotary_emb(q_state, kv_states, seqlen_offset=input_pos or 0)
            if input_pos is not None:
                kv_states = self.update_kv_cache(kv_states, input_pos)
            attn_output = self._sdpa(q_state, *kv_states.unbind(dim=2),
                                     attention_mask=attention_mask, is_causal=is_causal)

        attn_output = attn_output.contiguous().view(bsz, seqlen, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _sdpa(self, query_states: Tensor, key_states: Tensor, value_states: Tensor,
              attention_mask: Tensor, dropout: float = 0.0, is_causal: bool = False) -> Tensor:
        # transpose q, k, v for F.scaled_dot_product_attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        if attention_mask is not None:
            bsz, seqlen = query_states.size(0), query_states.size(2)
            kv_seq_len = key_states.size(2)
            if attention_mask.size() != (bsz, 1, seqlen, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seqlen, kv_seq_len)}, but is {attention_mask.size()}"
                )

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        return attn_output.transpose(1, 2)
