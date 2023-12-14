from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from flash_attn.ops.fused_dense import FusedDense

from model import ModelConfig


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.head_dim = config.hidden_size // config.num_attention_heads
        total_head_dim = (config.num_attention_heads + 2 * config.num_key_value_heads) * self.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = FusedDense(config.hidden_size, total_head_dim, bias=config.attention_bias)
        self.wo = FusedDense(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        self.num_heads = config.num_attention_heads
        self.n_local_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self._register_load_state_dict_pre_hook(self.load_hook)
        cache_shape = (config.max_batch_size, self.n_local_heads, config.max_position_embeddings, self.head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=torch.bfloat16, device='cuda'), persistent=False)
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=torch.bfloat16, device='cuda'), persistent=False)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    @torch.no_grad()
    def update_cache(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2], f"{input_pos.shape[0]} != {k_val.shape[2]}"

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.hidden_size, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if not self.training:
            k, v = self.update_cache(input_pos, k, v)

        k = k.repeat_interleave(self.num_heads // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)

        y = self.wo(y)
        return y
