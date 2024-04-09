from typing import Optional

import torch
from transformers import Cache

from .transformer import Transformer


class InternalCache(Cache):
    """
    Caching method that I cooked up for my own purposes.
    It's my way to work around huggingface's cache system.
    """

    def __init__(self, model: Transformer) -> None:
        super().__init__()
        self.model = model

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int | torch.Tensor:
        return self.model.layers[layer_idx].attention.get_cache_length()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer in self.model.layers:
            layer.attention.reorder_cache(beam_idx)
