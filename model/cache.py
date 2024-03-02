from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import Tensor
from transformers import Cache


class DynamicCache(Cache):
    """
    Same as huggingface's DynamicCache, but with a sequence_dim parameter to support different sequence dimensions.
    https://github.com/huggingface/transformers/blob/831bc25d8fdb85768402f772cf65cc3d7872b211/src/transformers/cache_utils.py#L61
    """
    def __init__(self, sequence_dim: int = 1) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = 0
        self.sequence_dim = sequence_dim

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        if layer_idx < len(self):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        raise IndexError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self) -> Tuple[Tensor, Tensor]:
        for layer_idx in range(len(self)):
            yield self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __len__(self):
        return len(self.key_cache)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[self.sequence_dim]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=self.sequence_dim)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=self.sequence_dim)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[self.sequence_dim]

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
