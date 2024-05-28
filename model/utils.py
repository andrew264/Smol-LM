# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
from typing import Union

from torch import nn

from .lora import LoRALinear

LINEAR = Union[nn.Linear, LoRALinear]


def hf_load_hook(state_dict: dict, prefix, *args, **kwargs):
    mappings = {
        "model.": "",
        ".self_attn.": ".attention_block.",
        ".mlp.": ".feed_forward.",
        ".block_sparse_moe.": ".feed_forward.",
    }

    def get_updated_key(key: str) -> str:
        updated_key = str(key)
        for old, new in mappings.items():
            updated_key = updated_key.replace(old, new)
        return updated_key

    for k in list(state_dict.keys()):
        state_dict[get_updated_key(k)] = state_dict.pop(k)
