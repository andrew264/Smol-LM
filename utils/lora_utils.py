from model import Transformer, LoRAConfig
from model.lora import LoRALinear


def to_lora_model(model: Transformer, config: LoRAConfig) -> Transformer:
    """
    Converts a transformer model to a LoRA model.
    :param model: (Transformer) The transformer model.
    :param config: (LoRAConfig) The LoRA configuration.
    :return: (nn.Module) The LoRA model.
    """
    lora_layers = config.lora_layers
    lora_dropout = config.lora_dropout
    lora_alpha = config.lora_alpha
    lora_r = config.lora_rank
    for param in model.parameters():
        param.requires_grad = False
    for block in model.layers:
        if 'qkv_proj' in lora_layers:
            block.attention.qkv_proj = LoRALinear(block.attention.qkv_proj, rank=lora_r, alpha=lora_alpha,
                                                  dropout=lora_dropout)
        if 'o_proj' in lora_layers:
            block.attention.o_proj = LoRALinear(block.attention.o_proj, rank=lora_r, alpha=lora_alpha,
                                                dropout=lora_dropout)
        if 'mlp' in lora_layers:
            ffn = block.feed_forward
            ffn.gate_proj = LoRALinear(ffn.gate_proj, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            ffn.up_proj = LoRALinear(ffn.up_proj, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            ffn.down_proj = LoRALinear(ffn.down_proj, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    return model
