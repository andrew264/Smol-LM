from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from .attention import AttentionBlock
from .config import ModelConfig
from .feed_forward import FeedForward
from .norm import get_rmsnorm_class
from .rotary import RotaryEmbedding


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = AttentionBlock(config, layer_idx=layer_idx)
        self.mlp = FeedForward(config)

        NORM_CLASS = get_rmsnorm_class()
        self.input_layernorm = NORM_CLASS(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NORM_CLASS(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: Tensor,
                freqs: Tensor,
                past_key_values: Optional[Cache] = None,
                attention_mask: Optional[Tensor] = None,
                cache_position: Optional[Tensor] = None, ) -> Tensor:
        # Self-attention
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x,
                           freqs=freqs,
                           past_key_values=past_key_values,
                           attention_mask=attention_mask,
                           cache_position=cache_position)

        residual = residual + x

        x = self.post_attention_layernorm(residual)
        x = self.mlp(x)
        x = residual + x

        return x


class TransformerBlocks(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(Block(config, idx) for idx in range(config.num_hidden_layers))
        self.norm = get_rmsnorm_class()(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(dim=config.hidden_size // config.num_attention_heads,
                                          base=config.rope_theta, )

    def forward(self,
                input_ids: Tensor,
                position_ids: Optional[Tensor] = None,
                cache_position: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                past_key_values: Optional[Cache] = None,
                ) -> Tensor:

        x = self.embed_tokens(input_ids)
        causal_mask = self._update_causal_mask(attention_mask, x, cache_position, past_key_values)
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = self.rotary_emb(position_ids)

        for i, layer in enumerate(self.layers):
            if self.training and i in self.config.checkpointing_layers:
                x = checkpoint(layer,
                               x, freqs, causal_mask,
                               use_reentrant=False, )

            else:
                x = layer(x,
                          freqs=freqs,
                          past_key_values=past_key_values,
                          attention_mask=causal_mask,
                          cache_position=cache_position, )

        x = self.norm(x)

        return x

    # copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    @staticmethod
    def _update_causal_mask(attention_mask: Optional[Tensor],
                            input_tensor: Tensor,
                            cache_position: Tensor,
                            past_key_values: Optional[Cache] = None):
        if attention_mask is None:
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = past_key_values.get_max_length() if past_key_values is not None else sequence_length

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask: Tensor = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        if attention_mask is not None and attention_mask.device.type == "cuda":
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
