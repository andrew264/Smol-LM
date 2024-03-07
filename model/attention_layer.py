from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from transformers import Cache

from model import ModelConfig
from model.rotary import RotaryEmbedding, apply_rotary_pos_emb


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

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
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(dim=self.head_dim,
                                          max_position_embeddings=self.max_position_embeddings,
                                          base=self.config.rope_theta, )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor],
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                ) -> Tensor:
        bsz, q_len, _ = hidden_states.size()
        is_causal = attention_mask is None and q_len > 1
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        attn_output = self._sdpa(query_states, key_states, value_states,
                                 attention_mask=attention_mask,
                                 dropout=self.attention_dropout,
                                 is_causal=is_causal)

        attn_output = attn_output.contiguous().view(bsz, q_len, self.hidden_size)
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
