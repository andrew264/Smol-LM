import os

import torch

from model import ModelConfig, Transformer


def load_model(config: ModelConfig, path: str, device: torch.device = torch.device('cuda')) -> torch.nn.Module:
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
        model.load_state_dict(torch.load(path, map_location=device))
        print("Loaded model from weights file.")
    else:
        print("Created new model.")
    model.eval()
    return model
