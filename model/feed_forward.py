import torch
import torch.nn as nn
from flash_attn.ops.fused_dense import FusedDense
from torch import Tensor
from torch.nn import functional as F
from transformers.activations import get_activation

from model import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = FusedDense(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = FusedDense(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = FusedDense(self.intermediate_size, self.hidden_size, bias=False)
        self.act = get_activation(config.hidden_act)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class SparseMoEBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        self.gate = FusedDense(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.num_experts)])

    def forward(self, x: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = x.shape
        x = x.reshape(-1, hidden_dim)
        router_logits = self.gate(x)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = routing_weights.topk(self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
