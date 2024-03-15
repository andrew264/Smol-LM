from typing import Union, Self

import torch
from torch import nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, dtype, device, dropout=.0):
        super().__init__()
        std_dev = 1. / (in_dim ** 0.5)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank, dtype=dtype, device=device) * std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim, dtype=dtype, device=device))

    def forward(self, x):
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.alpha


class LoRALinear(nn.Module):
    def __init__(self, linear: Union[nn.Linear, Self], rank, alpha, dropout=.0):
        super().__init__()
        if isinstance(linear, LoRALinear):
            linear = linear.linear
        self.linear = linear
        dtype = linear.weight.dtype
        device = linear.weight.device
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, dtype, device, dropout)

    def forward(self, x):
        return self.linear(x) + self.lora(x)
