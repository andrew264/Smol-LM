from typing import Union, Self

from torch import nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, dtype, device, dropout=.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features=in_dim, out_features=rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(in_features=rank, out_features=out_dim, bias=False, device=device, dtype=dtype)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        lora_out = self.lora_A(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_B(lora_out)
        return lora_out


class LoRALinear(nn.Module):
    def __init__(self, linear: Union[nn.Linear, Self], rank, alpha, dropout=.0):
        super().__init__()
        if isinstance(linear, LoRALinear):
            linear = linear.linear
        self.linear = linear
        self.linear.requires_grad = False
        dtype = linear.weight.dtype
        device = linear.weight.device
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, dtype, device, dropout)

    def forward(self, x):
        return self.linear(x) + self.lora(x)
