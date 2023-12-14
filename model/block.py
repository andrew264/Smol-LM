import torch.nn as nn
from flash_attn.ops.rms_norm import RMSNorm
from torch import Tensor

from model import ModelConfig
from model.attention_layer import Attention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
