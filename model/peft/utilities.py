import time
from typing import Optional, TypeVar

import torch
from torch import nn

from model.config import LoRAConfig
from model.peft import LoRALinear, LoRAEmbedding

T = TypeVar('T', bound=nn.Module)


def inject_lora_adapter(model: T,
                        lora_config: LoRAConfig,
                        adapter_state_dict: Optional[dict] = None,
                        merge_lora: bool = False) -> None:
    """
    Inject LoRA adapter into the model (inplace).
    """
    expected_layers = {'embedding', 'lm_head', 'qkv_proj', 'o_proj', 'mlp'}
    if not expected_layers.intersection(lora_config.layers):
        raise ValueError(f"Invalid layers to inject: {lora_config.layers}. Expected one of {expected_layers}")
    start = time.time()
    model.requires_grad_(False)

    LINEAR_CLS = LoRALinear
    EMBEDDING_CLS = LoRAEmbedding
    DEVICE = next(model.parameters()).device

    def inject_layer(target, attr_name, cls, config):
        if hasattr(target, attr_name): setattr(target, attr_name, cls(getattr(target, attr_name), config))

    if 'embedding' in lora_config.layers: inject_layer(model.model, 'embed_tokens', EMBEDDING_CLS, lora_config)

    if 'lm_head' in lora_config.layers: inject_layer(model, 'lm_head', LINEAR_CLS, lora_config)

    for layer in model.model.layers:
        block = layer.self_attn

        if 'qkv_proj' in lora_config.layers: inject_layer(block, 'qkv_proj', LINEAR_CLS, lora_config)

        if 'o_proj' in lora_config.layers: inject_layer(block, 'o_proj', LINEAR_CLS, lora_config)

        if 'mlp' in lora_config.layers:
            ffn = layer.mlp
            for proj in ['gate_proj', 'up_proj', 'down_proj']: inject_layer(ffn, proj, LINEAR_CLS, lora_config)

    model.to(DEVICE)
    print(f"LoRA injection took {time.time() - start:.3f}s")

    def linear_with_weight(weights: nn.Parameter, biases: Optional[nn.Parameter] = None):
        l = nn.Linear(weights.shape[1], weights.shape[0], bias=biases is not None).to(device=weights.device, dtype=weights.dtype)
        l.weight = nn.Parameter(weights)
        if biases is not None: l.bias = nn.Parameter(biases)
        return l

    def get_parent_and_module(name: str):
        parent_name = '.'.join(name.split('.')[:-1])
        module_name = name.split('.')[-1]
        parent = model
        for part in parent_name.split('.'):
            if part == '': continue
            parent = getattr(parent, part)
        return parent, module_name

    if adapter_state_dict is not None:
        start = time.time()
        model.load_state_dict(adapter_state_dict, strict=False, assign=True)
        if merge_lora:
            for name, module in model.named_modules():
                if isinstance(module, LINEAR_CLS):
                    setattr(*get_parent_and_module(name), linear_with_weight(*module.get_merged_weights()))
                elif isinstance(module, EMBEDDING_CLS):
                    setattr(*get_parent_and_module(name), nn.Embedding.from_pretrained(module.get_merged_weights(),
                                                                                       padding_idx=module.padding_idx,
                                                                                       sparse=module.sparse))
        print(f"Adapter state dict loading took {time.time() - start:.3f}s")


def get_lora_state_dict(sd: dict, dtype: torch.device = torch.bfloat16) -> dict:
    state_dict = {}
    for param_name, param in sd.items():
        if 'lora' in param_name: state_dict[param_name] = param.to(dtype=dtype)
    return state_dict
