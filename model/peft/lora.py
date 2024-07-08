from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from model.config import LoRAConfig


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, lora_config: LoRAConfig):
        super().__init__()
        self.linear = linear
        dtype = linear.weight.dtype
        device = linear.weight.device
        self.rank = lora_config.rank
        self.alpha = lora_config.alpha
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(lora_config.dropout) if lora_config.dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros((linear.in_features, self.rank), dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros((self.rank, linear.out_features), dtype=dtype, device=device))

        self.initialize_parameters()
        self._merged = False

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def get_merged_weights(self) -> Tuple[nn.Parameter, Optional[nn.Parameter]]:
        return (
            nn.Parameter(
                (self.lora_A @ self.lora_B).T * self.scaling + self.linear.weight,
                requires_grad=False
            ),
            nn.Parameter(self.linear.bias, requires_grad=False) if self.linear.bias is not None else None
        )

    def merge_weights(self):
        self.linear.weight = nn.Parameter(
            self.get_merged_weights()[0]
        )
        self.lora_A = None
        self.lora_B = None
        self._merged = True

    def lora_forward(self, x):
        dtype = x.dtype
        x = x.to(dtype=self.lora_A.dtype)
        x = self.scaling * (self.lora_dropout(x) @ self.lora_A @ self.lora_B)
        return x.to(dtype=dtype)

    def forward(self, x):
        if self._merged:
            return self.linear(x)
        return self.linear(x) + self.lora_forward(x)


class LoRAEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, lora_config: LoRAConfig):
        super().__init__()
        self.embedding = embedding
        dtype = embedding.weight.dtype
        device = embedding.weight.device
        self.padding_idx = embedding.padding_idx
        self.sparse = embedding.sparse
        self.rank = lora_config.rank
        self.alpha = lora_config.alpha
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.zeros((embedding.num_embeddings, self.rank), dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros((self.rank, embedding.embedding_dim), dtype=dtype, device=device))
        self.initialize_parameters()
        self._merged = False

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def get_merged_weights(self):
        return (self.lora_A @ self.lora_B) * self.scaling + self.embedding.weight

    def merge_weights(self):
        self.embedding.weight = nn.Parameter(
            self.get_merged_weights()
        )
        self.lora_A = None
        self.lora_B = None
        self._merged = True

    def lora_forward(self, x):
        return (self.scaling *
                F.embedding(x, self.lora_A, padding_idx=self.padding_idx, sparse=self.sparse) @ self.lora_B)

    def forward(self, x):
        if self._merged:
            return self.embedding(x)
        return self.embedding(x) + self.lora_forward(x)
