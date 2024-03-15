import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from transformers import StoppingCriteria

from model import ModelConfig, Transformer, LoRAConfig
from utils.lora_utils import to_lora_model


def load_model(config: ModelConfig, path: str,
               device: torch.device = torch.device('cuda:0')) -> Transformer:
    """
    Loads a model from a path.
    :param config: (ModelConfig) The model configuration.
    :param path: (str) The path to the model.
    :param device: (torch.device) The device to load the model to.
    :return: (torch.nn.Module) The model.
    """
    model = Transformer(config).to(dtype=torch.bfloat16, device=device)
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


def load_lora_model(config: ModelConfig, lora_config: LoRAConfig,
                    path: str, device: torch.device = torch.device('cuda:0')) -> Transformer:
    """
    Loads a LoRA model from a path.
    :param config: (ModelConfig) The model configuration.
    :param lora_config: (LoRAConfig) The LoRA configuration.
    :param path: (str) The path to the model.
    :param device: (torch.device) The device to load the model to.
    :return: (torch.nn.Module) The model.
    """
    model = Transformer(config)
    for param in model.parameters():
        param.requires_grad = False
    model = to_lora_model(model, lora_config).to(dtype=torch.bfloat16, device=device)
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
        print("Loaded optimizer from checkpoint.")
    else:
        print("Checkpoint file not found. Created new optimizer.")
    return optimizer


def save_optimizer(optimizer, path: str):
    """
    Saves an optimizer to a path.
    :param optimizer: (torch.optim.Optimizer) The optimizer to save.
    :param path: (str) The path to save the optimizer to.
    """
    torch.save(optimizer.state_dict(), path)


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=None, encounters=1):
        super().__init__()
        if stops is None:
            stops = []
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        stop_count = 0
        for stop in self.stops:
            if stop in input_ids[:, -1]:
                stop_count += 1

        return stop_count >= self.ENCOUNTERS


def count_parameters(model: torch.nn.Module):
    """
    Counts the number of parameters in a model.
    :param model: (torch.nn.Module) The model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params != total_trainable_params:
        print(f'{total_trainable_params:,} trainable parameters.')
        print(f'Percentage of trainable parameters: {total_trainable_params / total_params * 100:.2f}%')
