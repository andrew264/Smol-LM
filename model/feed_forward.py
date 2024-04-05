from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from transformers.activations import get_activation

from model import ModelConfig
from model.lora import LoRALinear


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj: Union[nn.Linear, LoRALinear] = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj: Union[nn.Linear, LoRALinear] = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj: Union[nn.Linear, LoRALinear] = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act = get_activation(config.hidden_act)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class ConditionalFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))
        self.act = get_activation(config.hidden_act)

        nn.init.normal_(self.gate_proj, std=config.initializer_range)
        nn.init.normal_(self.up_proj, std=config.initializer_range)
        nn.init.normal_(self.down_proj, std=config.initializer_range)

    def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
        gate_proj_weights = self.gate_proj[expert_indices]  # [T, A, D, D]
        down_proj_weights = self.down_proj[expert_indices]  # [T, A, D, D]
        up_proj_weights = self.up_proj[expert_indices]  # [T, A, D, D]
        x1 = self.act(torch.einsum('ti,taoi -> tao', x, gate_proj_weights))
        x3 = torch.einsum('ti, taoi -> tao', x, down_proj_weights)
        expert_outs = torch.einsum('tao, taio -> tai', (x1 * x3), up_proj_weights)
        return expert_outs


class SparseMoEBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.num_activated_experts = config.num_activated_experts

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = ConditionalFeedForward(config)

        self.jitter_noise = config.router_jitter_noise

    def forward(self, x: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = x.shape
        if self.training and self.jitter_noise > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        x = x.reshape(-1, hidden_dim)
        router_logits = self.gate(x)

        expert_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = expert_weights.topk(self.num_activated_experts, dim=-1)
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

        expert_outs = self.experts(x, expert_indices)
        return torch.einsum('tai,ta -> ti', expert_outs, expert_weights), expert_weights
