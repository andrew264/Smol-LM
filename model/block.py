from typing import Tuple, Optional

import torch
import torch.nn as nn
from flash_attn.ops.triton.layer_norm import RMSNorm
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import Cache

from model import ModelConfig
from model.attention_layer import Attention
from model.feed_forward import SparseMoEBlock, FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.layer_idx = layer_idx
        self.attention = Attention(config, layer_idx)
        self.is_moe = config.is_moe
        if self.is_moe:
            self.block_sparse_moe = SparseMoEBlock(config)
        else:
            self.feed_forward = FeedForward(config)

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: Tensor,
                attention_mask: Optional[Tensor],
                position_ids: Optional[int] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: Optional[bool] = False,
                output_router_logits: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Self-attention
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x, self_attn_weights, present_key_value = self.attention(x, attention_mask=attention_mask,
                                                                 position_ids=position_ids,
                                                                 past_key_value=past_key_value,
                                                                 output_attentions=output_attentions,
                                                                 use_cache=use_cache)
        x = residual + x

        # Block-sparse MoE
        residual = x
        router_logits = None
        x = self.post_attention_layernorm(x)
        if self.gradient_checkpointing and self.training:
            if self.is_moe:
                x, router_logits = checkpoint(self.block_sparse_moe.__call__, x, use_reentrant=False)
            else:
                x = checkpoint(self.feed_forward.__call__, x, use_reentrant=False)
        else:
            if self.is_moe:
                x, router_logits = self.block_sparse_moe(x)
            else:
                x = self.feed_forward(x)
        x = residual + x

        outputs = (x,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
