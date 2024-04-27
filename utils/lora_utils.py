import time
from typing import Optional

from model import SmolLM, LoRAConfig
from model.lora import LoRALinear


def inject_lora_adapter(model: SmolLM, lora_config: LoRAConfig, adapter_state_dict: Optional[dict] = None) -> SmolLM:
    start = time.time()
    for param in model.parameters():
        param.requires_grad = False
    if 'lm_head' in lora_config.layers and hasattr(model, 'lm_head'):
        model.lm_head = LoRALinear(model.lm_head, lora_config)
    for layer in model.layers:
        block = layer.jonkler_block
        # Attention
        if 'qkv_proj' in lora_config.layers and hasattr(block, 'qkv_proj'):
            block.qkv_proj = LoRALinear(block.qkv_proj, lora_config)
        if 'o_proj' in lora_config.layers and hasattr(block, 'o_proj'):
            block.o_proj = LoRALinear(block.o_proj, lora_config)
        # FeedForward
        if 'mlp' in lora_config.layers and hasattr(layer, 'feed_forward'):
            ffn = layer.feed_forward
            ffn.gate_proj = LoRALinear(ffn.gate_proj, lora_config)
            ffn.up_proj = LoRALinear(ffn.up_proj, lora_config)
            ffn.down_proj = LoRALinear(ffn.down_proj, lora_config)
        # Recurrent
        if 'l_x' in lora_config.layers and hasattr(block, 'linear_x'):
            block.linear_x = LoRALinear(block.linear_x, lora_config)
        if 'l_y' in lora_config.layers and hasattr(block, 'linear_y'):
            block.linear_y = LoRALinear(block.linear_y, lora_config)
        if 'l_out' in lora_config.layers and hasattr(block, 'linear_out'):
            block.linear_out = LoRALinear(block.linear_out, lora_config)

    print(f"LoRA injection took {time.time() - start:.3f}s")
    if adapter_state_dict is not None:
        start = time.time()
        model.load_state_dict(adapter_state_dict, strict=False)
        print(f"Adapter state dict loading took {time.time() - start:.3f}s")
    return model


def get_lora_state_dict(model: SmolLM) -> dict:
    state_dict = {}
    for param_name, param in model.named_parameters():
        if 'lora' in param_name:
            state_dict[param_name] = param
    return state_dict
