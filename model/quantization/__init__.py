import bitsandbytes as bnb
import torch.nn as nn


def replace_linear_with_linear8bitlt(module: nn.Module, fp16_weights: bool = False):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name,
                    bnb.nn.Linear8bitLt(child.in_features,
                                        child.out_features,
                                        bias=child.bias is not None,
                                        has_fp16_weights=fp16_weights))
        else:
            replace_linear_with_linear8bitlt(child, fp16_weights=fp16_weights)
