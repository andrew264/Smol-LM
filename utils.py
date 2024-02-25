import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file

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


def save_model(model: torch.nn.Module, path: str):
    """
    Saves a model to a path as a .safetensors file.
    :param model: (torch.nn.Module) The model to save.
    :param path: (str) The path to save the model to.
    """
    safe_save_file(model.state_dict(), path)


def load_optimizer(optimizer, path: str, device: torch.device = torch.device('cuda:0')):
    """
    Loads an optimizer from a path.
    :param optimizer: (torch.optim.Optimizer) The optimizer to load.
    :param path: (str) The path to the optimizer.
    :param device: (torch.device) The device to load the optimizer to.
    :return: (torch.optim.Optimizer) The optimizer.
    """
    if os.path.exists(path):
        optimizer.load_state_dict(torch.load(path, map_location=device))
        print("Loaded optimizer from weights file.")
    else:
        print("Weights file not found. Created new optimizer.")
    return optimizer


def save_optimizer(optimizer, path: str):
    """
    Saves an optimizer to a path.
    :param optimizer: (torch.optim.Optimizer) The optimizer to save.
    :param path: (str) The path to save the optimizer to.
    """
    torch.save(optimizer.state_dict(), path)
