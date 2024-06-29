import time
from typing import Optional, TypeVar

import torch
from torch import nn

from model import LoRAConfig
from model.lora import LoRALinear, LoRAEmbedding

T = TypeVar('T', bound=nn.Module)


def inject_lora_adapter(model: T,
                        lora_config: LoRAConfig,
                        adapter_state_dict: Optional[dict] = None,
                        merge_lora: bool = False) -> T:
    start = time.time()
    for param in model.parameters():
        param.requires_grad = False
    if 'embedding' in lora_config.layers and hasattr(model.model, 'embed_tokens'):
        model.model.embed_tokens = LoRAEmbedding(model.model.embed_tokens, lora_config)
    if 'lm_head' in lora_config.layers and hasattr(model, 'lm_head'):
        model.lm_head = LoRALinear(model.lm_head, lora_config)
    for layer in model.model.layers:
        block = layer.self_attn
        # Attention
        if 'qkv_proj' in lora_config.layers and hasattr(block, 'qkv_proj'):
            block.qkv_proj = LoRALinear(block.qkv_proj, lora_config)
        if 'o_proj' in lora_config.layers and hasattr(block, 'o_proj'):
            block.o_proj = LoRALinear(block.o_proj, lora_config)
        # FeedForward
        if 'mlp' in lora_config.layers and hasattr(layer, 'mlp'):
            ffn = layer.mlp
            ffn.gate_proj = LoRALinear(ffn.gate_proj, lora_config)
            ffn.up_proj = LoRALinear(ffn.up_proj, lora_config)
            ffn.down_proj = LoRALinear(ffn.down_proj, lora_config)

    torch.cuda.synchronize()
    print(f"LoRA injection took {time.time() - start:.3f}s")

    def linear_with_weight(weights: torch.Tensor):
        l = nn.Linear(weights.shape[1], weights.shape[0], device=weights.device, dtype=weights.dtype, bias=False)
        l.weight = nn.Parameter(weights)
        return l

    if adapter_state_dict is not None:
        start = time.time()
        model.load_state_dict(adapter_state_dict, strict=False)
        if merge_lora:
            for layer in model.model.layers:
                block = layer.self_attn
                if 'qkv_proj' in lora_config.layers and hasattr(block.qkv_proj, 'get_merged_weights'):
                    block.qkv_proj = linear_with_weight(block.qkv_proj.get_merged_weights())
                if 'o_proj' in lora_config.layers and hasattr(block.o_proj, 'get_merged_weights'):
                    block.o_proj = linear_with_weight(block.o_proj.get_merged_weights())
                if 'mlp' in lora_config.layers and hasattr(layer.mlp.gate_proj, 'get_merged_weights'):
                    layer.mlp.gate_proj = linear_with_weight(layer.mlp.gate_proj.get_merged_weights())
                    layer.mlp.up_proj = linear_with_weight(layer.mlp.up_proj.get_merged_weights())
                    layer.mlp.down_proj = linear_with_weight(layer.mlp.down_proj.get_merged_weights())
            if 'lm_head' in lora_config.layers and hasattr(model.lm_head, 'get_merged_weights'):
                model.lm_head = linear_with_weight(model.lm_head.get_merged_weights())
            if 'embedding' in lora_config.layers and hasattr(model.model.embed_tokens, 'get_merged_weights'):
                model.model.embed_tokens = nn.Embedding.from_pretrained(model.model.embed_tokens.get_merged_weights())
        torch.cuda.synchronize()
        print(f"Adapter state dict loading took {time.time() - start:.3f}s")
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    state_dict = {}
    for param_name, param in model.named_parameters():
        if 'lora' in param_name:
            state_dict[param_name] = param
    return state_dict
