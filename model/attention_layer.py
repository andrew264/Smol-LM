from typing import Optional

import torch
import torch.nn as nn
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

        cache_shape = (config.max_batch_size, config.max_position_embeddings, self.num_key_value_heads, self.head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=torch.bfloat16, device='cuda'), persistent=False)
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=torch.bfloat16, device='cuda'), persistent=False)

    @torch.no_grad()
    def update_cache(self, start_pos, k_val, v_val):
        seqlen = k_val.size(1)
        self.k_cache[:, start_pos: start_pos + seqlen] = k_val
        self.v_cache[:, start_pos: start_pos + seqlen] = v_val

        keys = self.k_cache[:, : start_pos + seqlen]
        values = self.v_cache[:, : start_pos + seqlen]

        return keys, values

    def forward(self, x: Tensor, mask: Tensor, freqs_cis: Tensor, start_pos: Optional[Tensor] = None, ) -> Tensor:
        bsz, seqlen, _ = x.size()

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = query_states.view(bsz, seqlen, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

        query_states, key_states = apply_rotary_emb(query_states, key_states, freqs_cis)

        if not self.training:
            key_states, value_states = self.update_cache(start_pos, key_states, value_states)

        query_states, key_states, value_states = map(lambda x: x.transpose(1, 2),
                                                     (query_states, key_states, value_states))

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states,
                                                     attn_mask=mask, dropout_p=0.0)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output
