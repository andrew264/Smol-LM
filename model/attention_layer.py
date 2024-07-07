from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import Cache

from .config import ModelConfig
from .rotary import apply_rotary_pos_emb


class AttentionBlock(nn.Module):
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
        self.key_value_hidden_size = self.num_key_value_heads * self.head_dim

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

        self.qkv_proj = nn.Linear(self.hidden_size,
                                  self.hidden_size + 2 * self.key_value_hidden_size,
                                  bias=config.attention_qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_out_bias)

        self._register_load_state_dict_pre_hook(self.fused_qkv_hook)

    @staticmethod
    def fused_qkv_hook(state_dict, prefix, *args, **kwargs):
        if prefix + 'q_proj.weight' in state_dict:
            q_weight = state_dict.pop(prefix + 'q_proj.weight')
            k_weight = state_dict.pop(prefix + 'k_proj.weight')
            v_weight = state_dict.pop(prefix + 'v_proj.weight')
            state_dict[prefix + 'qkv_proj.weight'] = torch.cat([q_weight, k_weight, v_weight])
        if prefix + 'q_proj.bias' in state_dict:
            q_bias = state_dict.pop(prefix + 'q_proj.bias')
            k_bias = state_dict.pop(prefix + 'k_proj.bias')
            v_bias = state_dict.pop(prefix + 'v_proj.bias')
            state_dict[prefix + 'qkv_proj.bias'] = torch.cat([q_bias, k_bias, v_bias])

    def forward(self,
                hidden_states: Tensor,
                freqs: Tensor,
                past_key_values: Optional[Cache] = None,
                attention_mask: Optional[Tensor] = None,
                cache_position: Optional[Tensor] = None,
                ) -> Tensor:
        bsz, seqlen, _ = hidden_states.size()
        is_causal = attention_mask is None and seqlen > 1

        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.split(
            [
                self.hidden_size,
                self.key_value_hidden_size,
                self.key_value_hidden_size,
            ],
            dim=2,
        )
        query_states = query_states.view(bsz, seqlen, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *freqs)

        if cache_position is not None and past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx,
                                                              dict(cache_position=cache_position))

        attn_output = sdpa(query_states, key_states, value_states,
                           attention_mask=attention_mask,
                           num_key_value_groups=self.num_key_value_groups,
                           dropout=self.attention_dropout if self.training else 0.0,
                           is_causal=is_causal)
        attn_output = attn_output.contiguous().view(bsz, seqlen, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


def sdpa(query_states: Tensor,
         key_states: Tensor,
         value_states: Tensor,
         attention_mask: Optional[Tensor] = None,
         num_key_value_groups: int = 1,
         dropout: float = 0.0,
         is_causal: bool = False) -> Tensor:
    # transpose q, k, v for scaled_dot_product_attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if num_key_value_groups > 1:
        key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, :key_states.shape[2]]

    attn_output = nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=is_causal,
    )
    return attn_output.transpose(1, 2)
