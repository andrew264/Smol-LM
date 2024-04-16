import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
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


class SparseMoEBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.num_activated_experts = config.num_activated_experts

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.num_experts)])

        self.jitter_noise = config.router_jitter_noise

    def forward(self, x: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = x.shape
        if self.training and self.jitter_noise > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        x = x.reshape(-1, hidden_dim)
        router_logits = self.gate(x)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = routing_weights.topk(self.num_activated_experts, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = x[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits
