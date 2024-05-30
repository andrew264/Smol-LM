from typing import Union, Optional, List

import torch
from torch import nn, Tensor

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


def merge_audio_features(input_embeds: Tensor,
                         attention_mask: Optional[Tensor],
                         labels: Optional[Tensor],
                         audio_features: Tensor,
                         max_length: int,
                         ):
    device = input_embeds.device
    if labels is not None:
        label_pad = torch.full(
            (audio_features.shape[0], audio_features.shape[1]),
            -100,
            dtype=torch.long,
            device=device
        )
        labels = torch.cat((label_pad, labels), dim=1)

    if attention_mask is not None:
        attention_pad = torch.ones(
            (audio_features.shape[0], audio_features.shape[1]),
            dtype=torch.long,
            device=device
        )
        attention_mask = torch.cat((attention_pad, attention_mask), dim=1)

    combined_features = torch.cat((audio_features, input_embeds), dim=1)

    if combined_features.shape[1] > max_length:
        combined_features = combined_features[:, -max_length:]

    if labels is not None:
        truncated_labels = labels[:, -max_length:] if labels.shape[1] > max_length else labels
    else:
        truncated_labels = None

    return combined_features, attention_mask, truncated_labels


def get_optimizer_grouped_parameters(model: nn.Module, weight_decay: float) -> list[dict]:  # from llm.c repo
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    return optim_groups


def get_lora_plus_optimizer_group(model: nn.Module,
                                  lr: float,
                                  lr_ratio: int = 4,
                                  lr_embedding: float = 1e-6,
                                  ) -> List[dict]:
    param_groups = {
        "groupA": {},
        "groupB": {},
        "embedding": {},
    }
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'tok_embeddings' in param_name:
            param_groups["embedding"][param_name] = param
        elif 'lora_A' in param_name:
            param_groups["groupA"][param_name] = param
        elif 'lora_B' in param_name:
            param_groups["groupB"][param_name] = param

    optimizer_grouped_parameters = [
        {"params": param_groups["groupA"].values(), "lr": lr},
        {"params": param_groups["groupB"].values(), "lr": lr * lr_ratio},  # learn the B group faster than A
        {"params": param_groups["embedding"].values(), "lr": lr_embedding},
    ]
    return optimizer_grouped_parameters
