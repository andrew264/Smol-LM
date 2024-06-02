import torch.nn as nn
from torch import Tensor
from transformers.activations import get_activation

from .config import ModelConfig
from .utils import LINEAR


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj: LINEAR = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj: LINEAR = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj: LINEAR = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act = get_activation(config.hidden_act)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
