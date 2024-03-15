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
    for block in model.layers:
        if 'q_proj' in lora_layers:
            block.attention.q_proj = LoRALinear(block.attention.q_proj, rank=lora_r, alpha=lora_alpha,
                                                dropout=lora_dropout)
            block.attention.q_proj.requires_grad = True
        if 'k_proj' in lora_layers:
            block.attention.k_proj = LoRALinear(block.attention.k_proj, rank=lora_r, alpha=lora_alpha,
                                                dropout=lora_dropout)
            block.attention.k_proj.requires_grad = True
        if 'v_proj' in lora_layers:
            block.attention.v_proj = LoRALinear(block.attention.v_proj, rank=lora_r, alpha=lora_alpha,
                                                dropout=lora_dropout)
            block.attention.v_proj.requires_grad = True
        if 'o_proj' in lora_layers:
            block.attention.o_proj = LoRALinear(block.attention.o_proj, rank=lora_r, alpha=lora_alpha,
                                                dropout=lora_dropout)
            block.attention.o_proj.requires_grad = True
        if 'mlp' in lora_layers:
            ffn = block.feed_forward
            ffn.gate_proj = LoRALinear(ffn.gate_proj, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            ffn.up_proj = LoRALinear(ffn.up_proj, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            ffn.down_proj = LoRALinear(ffn.down_proj, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            ffn.requires_grad = True

    return model
