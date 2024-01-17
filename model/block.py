from typing import Tuple, Optional

import torch.nn as nn
from flash_attn.ops.rms_norm import RMSNorm
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from model import ModelConfig
from model.attention_layer import Attention
from model.feed_forward import SparseMoEBlock, FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.attention = Attention(config)
        self.is_moe = config.is_moe
        if self.is_moe:
            self.block_sparse_moe = SparseMoEBlock(config)
        else:
            self.feed_forward = FeedForward(config)

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: Tensor, mask: Optional[Tensor], input_pos: Optional[int]) -> Tuple[Tensor, Tensor]:
        # Self-attention
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.attention(x, mask, input_pos=input_pos)

        # Block-sparse MoE
        residual = x
        router_logits = None
        x = self.post_attention_layernorm(x)
        if self.gradient_checkpointing and self.training:
            if self.is_moe:
                x, router_logits = checkpoint(self.block_sparse_moe.__call__, x, use_reentrant=False)
            else:
                x = checkpoint(self.feed_forward.__call__, x, use_reentrant=False)
        else:
            if self.is_moe:
                x, router_logits = self.block_sparse_moe(x)
            else:
                x = self.feed_forward(x)
        x = residual + x

        return x, router_logits
