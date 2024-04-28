from typing import Union, Self

import torch
from torch import nn
from torch.nn import functional as F

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
        self.scaling = self.alpha / self.rank
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
        return self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))

    def forward(self, x):
        return self.linear(x) + self.lora_forward(x)


class LoRAEmbedding(nn.Module):
    def __init__(self, embedding: Union[nn.Embedding, Self], lora_config: LoRAConfig):
        super().__init__()
        if isinstance(embedding, LoRAEmbedding):
            embedding = embedding.embedding
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

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def lora_forward(self, x):
        return (self.scaling *
                F.embedding(x, self.lora_A, padding_idx=self.padding_idx, sparse=self.sparse) @ self.lora_B)

    def forward(self, x):
        return self.embedding(x) + self.lora_forward(x)
