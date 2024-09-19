from typing import Optional, List, Dict, Any, Tuple

import torch
from torch import Tensor
from transformers import Cache

from model.config import ModelConfig


class StaticCache(Cache):
    """
    same as transformers's StaticCache but with different cache_shape
    """

    def __init__(self, config: ModelConfig,
                 compiled_mode: bool = False,
                 dtype: torch.dtype = torch.bfloat16,
                 device: Optional[torch.device] = None) -> None:
        self.config = config
        self.is_compiled = compiled_mode
        self.dtype = dtype
        
        if device: self.device = device
        elif torch.cuda.is_available(): self.device = torch.device('cuda')
        else: self.device = torch.device('cpu')

        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        cache_shape = (
            config.max_batch_size,
            config.max_position_embeddings,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device=self.device)
        for _ in range(config.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            if self.is_compiled:
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def get_seq_length(self, layer_idx: Optional[int] = -1) -> int | torch.Tensor:
        return (self.key_cache[layer_idx][0, :, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        return self.config.max_position_embeddings

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = key_states.shape[0]
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        k_out[:bsz, cache_position, :] = key_states
        v_out[:bsz, cache_position, :] = value_states

        if self.is_compiled:
            return k_out, v_out
        else:
            last_position = cache_position[-1] + 1
            return k_out[:bsz, :last_position], v_out[:bsz, :last_position]

    def reset(self):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()
