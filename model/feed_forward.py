import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from flash_attn.ops.fused_dense import FusedDense

from model import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.w1 = FusedDense(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = FusedDense(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = FusedDense(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
