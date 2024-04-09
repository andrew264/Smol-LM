from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import Tensor
from transformers import Cache

from .config import ModelConfig
from .transformer import Transformer


class DynamicCache(Cache):
    """
    Same as huggingface's DynamicCache, but with a sequence_dim parameter to support different sequence dimensions.
    https://github.com/huggingface/transformers/blob/831bc25d8fdb85768402f772cf65cc3d7872b211/src/transformers/cache_utils.py#L61
    """

    def __init__(self, sequence_dim: int = 1) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0
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
            self._seen_tokens += key_states.shape[self.sequence_dim]

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


class StaticCache(Cache):
    """
    Not Same as huggingface's StaticCache, with sequence_dim set to 1
    https://github.com/huggingface/transformers/blob/e9c23fa056f401a586a1691edf773d1b9b60f96d/src/transformers/cache_utils.py#L344
    """

    def __init__(
            self,
            config: ModelConfig,
            max_batch_size: Optional[int] = None,
            max_cache_len: Optional[int] = None,
            device=torch.device("cuda"),
            dtype=None
    ) -> None:
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.max_batch_size = max_batch_size if max_batch_size is not None else config.max_batch_size
        self.max_cache_len = max_cache_len if max_cache_len is not None else config.max_position_embeddings
        self.device = device
        self.dtype = dtype if dtype is not None else torch.bfloat16

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        cache_shape = (
            self.num_layers,
            self.max_batch_size,
            self.max_cache_len,
            self.num_key_value_heads,
            self.head_dim
        )

        self.key_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        self.value_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_cache_positions = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        last_position = new_cache_positions[-1] + 1

        k_out[:, new_cache_positions] = key_states
        v_out[:, new_cache_positions] = value_states
        return k_out[:, :last_position], v_out[:, :last_position]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int | torch.Tensor:
        return (self.key_cache[0, 0, :, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        return self.max_cache_len

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        device = self.key_cache.device
        self.key_cache = self.key_cache.index_select(1, beam_idx.to(device))
        device = self.value_cache.device
        self.value_cache = self.value_cache.index_select(1, beam_idx.to(device))


class InternalCache(Cache):
    """
    Caching method that I cooked up for my own purposes.
    It's my way to work around huggerface's cache system.
    """

    def __init__(self, model: Transformer) -> None:
        super().__init__()
        self.model = model

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int | torch.Tensor:
        return self.model.layers[layer_idx].attention.get_cache_length()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer in self.model.layers:
            layer.attention.reorder_cache(beam_idx)
