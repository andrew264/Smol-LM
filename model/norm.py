from typing import Type

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = self._norm(hidden_states.float()) * self.weight
        return hidden_states.to(input_dtype)


def get_rmsnorm_class() -> Type[nn.Module]:
    try:
        from flash_attn.ops.rms_norm import RMSNorm as FlashRMSNorm
        return FlashRMSNorm
    except ImportError:
        try:
            from torch.nn import RMSNorm as TorchRMSNorm
            return TorchRMSNorm
        except ImportError:
            return RMSNorm
