from typing import Tuple, Optional, Union

import torch.nn as nn
from torch import Tensor

from .attention_layer import AttentionBlock
from .config import ModelConfig
from .feed_forward import SparseMoEBlock, FeedForward
from .recurrent_layer import RecurrentBlock

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
    from .norm import RMSNorm

BLOCK_CLASSES = {
    "attention": AttentionBlock,
    "recurrent": RecurrentBlock,
}
J_BLOCK = Union[AttentionBlock, RecurrentBlock]
F_BLOCK = Union[SparseMoEBlock, FeedForward]


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # jonkler; deal with it
        self.jonkler_block: J_BLOCK = BLOCK_CLASSES[config.layers_block_types[layer_idx]](config)

        self.is_moe = config.is_moe
        self.feed_forward: F_BLOCK = SparseMoEBlock(config) if self.is_moe else FeedForward(config)

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: Tensor,
                attention_mask: Optional[Tensor],
                position_ids: Optional[Tensor] = None,
                cache_position: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        # Self-attention
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.jonkler_block(x,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               cache_position=cache_position)
        x = residual + x

        # Block-sparse MoE
        residual = x
        router_logits = None
        x = self.post_attention_layernorm(x)
        if self.is_moe:
            x, router_logits = self.feed_forward(x)
        else:
            x = self.feed_forward(x)
        output = residual + x

        return output, router_logits
