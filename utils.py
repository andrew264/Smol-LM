import os

import torch
from safetensors import safe_open

from model import ModelConfig, Transformer


def load_model(config: ModelConfig, path: str, device: torch.device = torch.device('cuda:0')) -> torch.nn.Module:
    """
    Loads a model from a path.
    :param config: (ModelConfig) The model configuration.
    :param path: (str) The path to the model.
    :param device: (torch.device) The device to load the model to.
    :return: (torch.nn.Module) The model.
    """
    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)
    if os.path.exists(path):
        state_dict = {}
        d = device.type if device.type == 'cpu' else device.index
        with safe_open(path, framework="pt", device=d) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        model.load_state_dict(state_dict)
        print("Loaded model from weights file.")
        del state_dict
        torch.cuda.empty_cache()
    else:
        print("Created new model.")
    model.eval()
    return model
