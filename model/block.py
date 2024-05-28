from typing import Tuple, Optional

import torch.nn as nn
from torch import Tensor

from .attention_layer import AttentionBlock
from .config import ModelConfig
from .feed_forward import FeedForward
from .norm import get_rmsnorm_class


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_block = AttentionBlock(config)
        self.feed_forward = FeedForward(config)

        NORM_CLASS = get_rmsnorm_class()
        self.input_layernorm = NORM_CLASS(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NORM_CLASS(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None,
                cache_position: Optional[Tensor] = None, ) -> Tuple[Tensor, Optional[Tensor]]:
        # Self-attention
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.attention_block(x,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 cache_position=cache_position)

        residual = residual + x

        x = self.post_attention_layernorm(residual)
        x = self.feed_forward(x)
        x = residual + x

        return x
