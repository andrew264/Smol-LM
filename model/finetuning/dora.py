from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from model.config import LoRAConfig


class DoRALinear(nn.Module):
    # Weight-Decomposed Low-Rank Adaptation
    # https://github.com/catid/dora/blob/main/dora.py
    def __init__(self, linear: nn.Linear, lora_config: LoRAConfig):
        super().__init__()
        self.weight = linear.weight
        self.weight.requires_grad = False
        self.bias = linear.bias
        if self.bias is not None:
            self.bias.requires_grad = False
        self.rank = lora_config.rank

        # Initialize LoRA weights
        d_out, d_in = linear.weight.shape
        std_dev = 1 / torch.sqrt(torch.tensor(self.rank).float())
        self.lora_A = nn.Parameter(torch.randn(d_out, self.rank) * std_dev)
        self.lora_B = nn.Parameter(torch.zeros(self.rank, d_in))

        # Initialize the magnitude parameter
        self.lora_magnitude = nn.Parameter(self.weight.norm(p=2, dim=1, keepdim=True))

        self._merged = False

    def get_merged_weights(self, scaling: float | torch.Tensor = 1.0) -> Tuple[nn.Parameter, Optional[nn.Parameter]]:
        lora_weight = self.lora_A @ self.lora_B
        adapted = self.weight + scaling * lora_weight

        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adopted = adapted / column_norm
        return (nn.Parameter(norm_adopted * self.lora_magnitude, requires_grad=False),
                nn.Parameter(self.bias, requires_grad=False) if self.bias is not None else None)

    def merge_weights(self, scaling: float | torch.Tensor = 1.0):
        self.weight, self.bias = self.get_merged_weights(scaling)
        self.lora_A = None
        self.lora_B = None
        self.lora_magnitude = None
        self._merged = True

    def dora_forward(self, x: torch.Tensor, scaling: float | torch.Tensor = 1.0) -> torch.Tensor:
        adapted = self.weight + scaling * (self.lora_A @ self.lora_B)
        norm_adopted = adapted / adapted.norm(p=2, dim=0, keepdim=True)
        return F.linear(x, norm_adopted * self.lora_magnitude, self.bias)

    def forward(self, x: torch.Tensor, scaling: float | torch.Tensor = 1.0) -> torch.Tensor:
        if self._merged:
            return F.linear(x, self.weight, self.bias)
        return self.dora_forward(x, scaling)


class DoRAEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, lora_config: LoRAConfig):
        super().__init__()
        self.weight = embedding.weight
        self.weight.requires_grad = False
        self.padding_idx = embedding.padding_idx
        self.sparse = embedding.sparse

        self.rank = lora_config.rank

        # Initialize LoRA weights
        d_out, d_in = embedding.num_embeddings, embedding.embedding_dim
        std_dev = 1 / torch.sqrt(torch.tensor(self.rank).float())
        self.lora_A = nn.Parameter(torch.randn(d_out, self.rank) * std_dev)
        self.lora_B = nn.Parameter(torch.zeros(self.rank, d_in))

        # Initialize the magnitude parameter
        self.lora_magnitude = nn.Parameter(self.weight.norm(p=2, dim=1, keepdim=True))

        self._merged = False

    def get_merged_weights(self, scaling: float | torch.Tensor = 1.0) -> nn.Parameter:
        lora_weight = self.lora_A @ self.lora_B
        adapted = self.weight + scaling * lora_weight

        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adopted = adapted / column_norm
        return nn.Parameter(norm_adopted * self.lora_magnitude, requires_grad=False)

    def merge_weights(self, scaling: float | torch.Tensor = 1.0):
        self.weight = self.get_merged_weights(scaling)
        self.lora_A = None
        self.lora_B = None
        self.lora_magnitude = None
        self._merged = True

    def dora_forward(self, x: torch.Tensor, scaling: float | torch.Tensor = 1.0) -> torch.Tensor:
        adapted = self.weight + scaling * self.lora_A @ self.lora_B
        norm_adopted = adapted / adapted.norm(p=2, dim=0, keepdim=True)

        return F.embedding(x, norm_adopted * self.lora_magnitude, padding_idx=self.padding_idx, sparse=self.sparse)

    def forward(self, x: torch.Tensor, scaling: float | torch.Tensor = 1.0) -> torch.Tensor:
        if self._merged:
            return F.embedding(x, self.weight, padding_idx=self.padding_idx, sparse=self.sparse)
        return self.dora_forward(x, scaling)
