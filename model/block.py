from typing import Tuple

import torch.nn as nn
from flash_attn.ops.rms_norm import RMSNorm
from torch import Tensor

from model import ModelConfig
from model.attention_layer import Attention
from model.feed_forward import SparseMoEBlock


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.block_sparse_moe = SparseMoEBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: Tensor, start_pos: Tensor, mask: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
        # Self-attention
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.attention(x, mask, start_pos=start_pos, freqs_cis=freqs_cis)

        # Block-sparse MoE
        residual = x
        x = self.post_attention_layernorm(x)
        x, router_logits = self.block_sparse_moe(x)
        x = residual + x

        return x, router_logits
