from typing import Union, Optional

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


def get_optimizer_grouped_parameters(model: nn.Module, weight_decay) -> list[dict]:  # from nano gpt repo
    decay_denylist = ["embed_tokens.weight"]
    # start with all the candidate parameters
    decay = set()
    no_decay = set()
    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = param
        if param.ndimension() == 1 or any(nd in name for nd in decay_denylist):
            no_decay.add(name)
        else:
            decay.add(name)

    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params),)

    optim_groups = [
        {'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0},
        {'params': [param_dict[pn] for pn in sorted(list(decay))], 'weight_decay': weight_decay},
    ]

    return optim_groups
