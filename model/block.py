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

    def forward(self, x: Tensor, start_pos: Tensor, mask: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
        # Self-attention
        residual = x
        x = self.input_layernorm(x)
        attn_outputs = self.attention(x, mask, start_pos=start_pos, freqs_cis=freqs_cis)

        # Block-sparse MoE
        feed_forward_hidden_states, router_logits = self.block_sparse_moe(x)
        x = residual + attn_outputs + feed_forward_hidden_states

        return x, router_logits
