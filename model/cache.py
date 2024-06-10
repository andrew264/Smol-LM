from typing import Optional

import torch
from transformers import Cache

from .transformer import SmolLM


class InternalCache(Cache):
    """
    Caching method that I cooked up for my own purposes.
    It's my way to work around huggingface's cache system.
    """

    def __init__(self, model: SmolLM, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.reset_cache()

    def get_seq_length(self, layer_idx: Optional[int] = -1) -> int | torch.Tensor:
        return self.model.model.layers[layer_idx].self_attn.get_cache_length()

    def reset_cache(self):
        for layer in self.model.model.layers:
            layer.self_attn.setup_cache(dtype=self.dtype, device=self.model.device)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer in self.model.model.layers:
            layer.self_attn.reorder_cache(beam_idx)
