from model import SmolLM, LoRAConfig
from model.lora import LoRALinear


def to_lora_model(model: SmolLM, config: LoRAConfig) -> SmolLM:
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
        if hasattr(block.jonkler_block, 'qkv_proj') and 'qkv_proj' in lora_layers:
            block.jonkler_block.qkv_proj = LoRALinear(block.jonkler_block.qkv_proj, rank=lora_r, alpha=lora_alpha,
                                                      dropout=lora_dropout)
        if hasattr(block.jonkler_block, 'o_proj') and 'o_proj' in lora_layers:
            block.jonkler_block.o_proj = LoRALinear(block.jonkler_block.o_proj, rank=lora_r, alpha=lora_alpha,
                                                    dropout=lora_dropout)
        if hasattr(block.jonkler_block, 'linear_x') and 'l_x' in lora_layers:
            block.jonkler_block.linear_x = LoRALinear(block.jonkler_block.linear_x, rank=lora_r, alpha=lora_alpha,
                                                      dropout=lora_dropout)
        if hasattr(block.jonkler_block, 'linear_y') and 'l_y' in lora_layers:
            block.jonkler_block.linear_y = LoRALinear(block.jonkler_block.linear_y, rank=lora_r, alpha=lora_alpha,
                                                      dropout=lora_dropout)
        if hasattr(block.jonkler_block, 'linear_out') and 'l_out' in lora_layers:
            block.jonkler_block.linear_out = LoRALinear(block.jonkler_block.linear_out, rank=lora_r, alpha=lora_alpha,
                                                        dropout=lora_dropout)
        if 'mlp' in lora_layers:
            ffn = block.feed_forward
            ffn.gate_proj = LoRALinear(ffn.gate_proj, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            ffn.up_proj = LoRALinear(ffn.up_proj, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            ffn.down_proj = LoRALinear(ffn.down_proj, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    return model
