from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .config import ModelConfig
from .lora import LoRALinear
from .rotary import RotaryEmbedding, apply_rotary_pos_emb


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

        self.qkv_proj: Union[nn.Linear, LoRALinear] = nn.Linear(
            self.hidden_size,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)
        self.o_proj: Union[nn.Linear, LoRALinear] = nn.Linear(self.hidden_size, self.hidden_size,
                                                              bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(dim=self.head_dim,
                                          max_position_embeddings=self.max_position_embeddings,
                                          base=self.config.rope_theta, )
        self._register_load_state_dict_pre_hook(self.fused_qkv_hook)

        # kv cache
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None

    @staticmethod
    def fused_qkv_hook(state_dict, prefix, *args, **kwargs):
        if prefix + 'q_proj.weight' in state_dict:
            q_weight = state_dict.pop(prefix + 'q_proj.weight')
            k_weight = state_dict.pop(prefix + 'k_proj.weight')
            v_weight = state_dict.pop(prefix + 'v_proj.weight')
            state_dict[prefix + 'qkv_proj.weight'] = torch.cat([q_weight, k_weight, v_weight])

    def _setup_cache(self, dtype: torch.dtype, device: torch.device):
        cache_shape = (
            self.config.max_batch_size,
            self.max_position_embeddings,
            self.num_key_value_heads,
            self.head_dim
        )
        self.key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        if self.layer_idx == 0:
            print(f"KV Cache initialized with shape: {cache_shape}")

    @torch.no_grad()
    def _update_cache(self, key_states: Tensor, value_states: Tensor, cache_position: Tensor) -> Tuple[Tensor, Tensor]:
        if self.key_cache is None or self.value_cache is None:
            self._setup_cache(dtype=key_states.dtype, device=key_states.device)

        last_position = cache_position[-1] + 1
        self.key_cache[:, cache_position] = key_states
        self.value_cache[:, cache_position] = value_states
        return self.key_cache[:, :last_position], self.value_cache[:, :last_position]

    @torch.no_grad()
    def reorder_cache(self, beam_idx: Tensor) -> None:
        assert self.key_cache is not None, "Cache is not initialized, call _setup_cache() first."
        self.key_cache = self.key_cache.index_select(0, beam_idx.to(self.key_cache.device))
        self.value_cache = self.value_cache.index_select(0, beam_idx.to(self.value_cache.device))

    def get_cache_length(self) -> int | Tensor:
        if self.key_cache is None:
            return 0
        return (self.key_cache[0, :, 0].any(dim=-1)).sum()

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor],
                position_ids: Optional[torch.LongTensor] = None,
                **kwargs,
                ) -> Tensor:
        bsz, seqlen, _ = hidden_states.size()
        is_causal = attention_mask is None and seqlen > 1
        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.split(
            [
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=2,
        )
        query_states = query_states.view(bsz, seqlen, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        cache_position = kwargs.get("cache_position", None)
        if cache_position is not None and not self.training:
            key_states, value_states = self._update_cache(key_states, value_states, cache_position)

        attn_output = self._sdpa(query_states, key_states, value_states,
                                 attention_mask=attention_mask,
                                 dropout=self.attention_dropout,
                                 is_causal=is_causal)

        attn_output = attn_output.contiguous().view(bsz, seqlen, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _sdpa(self, query_states: Tensor, key_states: Tensor, value_states: Tensor,
              attention_mask: Tensor, dropout: float = 0.0, is_causal: bool = False) -> Tensor:
        # transpose q, k, v for F.scaled_dot_product_attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.num_key_value_heads > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[2]]

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        return attn_output.transpose(1, 2)
