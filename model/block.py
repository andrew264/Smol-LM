from typing import Optional

import torch.nn as nn
from torch import Tensor

from .attention_layer import AttentionBlock
from .config import ModelConfig
from .feed_forward import FeedForward
from .norm import get_rmsnorm_class


class Block(nn.Module):
    def __init__(self, config: ModelConfig, causal: Optional[bool] = None) -> None:
        super().__init__()
        self.self_attn = AttentionBlock(config, causal=causal)
        self.mlp = FeedForward(config)

        NORM_CLASS = get_rmsnorm_class()
        self.input_layernorm = NORM_CLASS(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NORM_CLASS(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: Tensor,
                freqs: Tensor,
                attention_mask: Optional[Tensor] = None,
                cache_position: Optional[Tensor] = None, ) -> Tensor:
        # Self-attention
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x,
                           freqs=freqs,
                           attention_mask=attention_mask,
                           cache_position=cache_position)

        residual = residual + x

        x = self.post_attention_layernorm(residual)
        x = self.mlp(x)
        x = residual + x

        return x
