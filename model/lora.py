from typing import Union, Self

from torch import nn

from model.config import LoRAConfig


class LoRALinear(nn.Module):
    def __init__(self, linear: Union[nn.Linear, Self], lora_config: LoRAConfig):
        super().__init__()
        if isinstance(linear, LoRALinear):
            linear = linear.linear
        self.linear = linear
        dtype = linear.weight.dtype
        device = linear.weight.device
        self.rank = lora_config.rank
        self.alpha = lora_config.alpha
        self.alpha_by_rank = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(lora_config.dropout) if lora_config.dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features=linear.in_features, out_features=self.rank,
                                bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(in_features=self.rank, out_features=linear.out_features,
                                bias=False, device=device, dtype=dtype)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def lora_forward(self, x):
        return self.alpha_by_rank * self.lora_B(self.lora_A(self.lora_dropout(x)))

    def forward(self, x):
        return self.linear(x) + self.lora_forward(x)
