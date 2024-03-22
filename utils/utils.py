import os
from typing import Optional, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import StoppingCriteria

from model import ModelConfig, Transformer, LoRAConfig
from utils.lora_utils import to_lora_model


def load_model(config: ModelConfig, lora_config: Optional[LoRAConfig] = None,
               path: str = None, device: torch.device = torch.device('cpu')) -> Transformer:
    state_dict = {}
    if path and os.path.exists(path):
        d = device.type if device.type == 'cpu' else device.index
        with safe_open(path, framework="pt", device=d) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        print("Loaded model from weights file.")
    else:
        print("Created new model.")

    is_lora_state = any('lora' in k for k in state_dict.keys())
    model = Transformer(config)
    if is_lora_state:
        assert lora_config is not None, "LoRA config must be provided if model weights have LoRA."
        model = to_lora_model(model, lora_config)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
        if lora_config is not None:
            model = to_lora_model(model, lora_config)
    del state_dict
    torch.cuda.empty_cache()
    model.to(dtype=torch.bfloat16, device=device)
    model.eval()
    return model


def save_model(model: torch.nn.Module, path: str):
    """
    Saves a model to a path as a .safetensors file.
    :param model: (torch.nn.Module) The model to save.
    :param path: (str) The path to save the model to.
    """
    safe_save_file(model.state_dict(), path)


def load_optimizer(optimizer: Optimizer, path: str, device: torch.device = torch.device('cuda:0')):
    """
    Loads an optimizer from a path.
    :param optimizer: (torch.optim.Optimizer) The optimizer to load.
    :param path: (str) The path to the optimizer.
    :param device: (torch.device) The device to load the optimizer to.
    :return: (torch.optim.Optimizer) The optimizer.
    """
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        try:
            optimizer.load_state_dict(state_dict)
            print("Loaded optimizer from checkpoint.")
        except ValueError as e:
            print(f"Error loading optimizer: {e}", "Creating new optimizer.")
        del state_dict
        torch.cuda.empty_cache()
    else:
        print("Optimizer checkpoint file not found.")
    return optimizer


def load_scheduler(scheduler: LRScheduler, path: str, device: torch.device = torch.device('cuda:0')):
    if os.path.exists(path):
        scheduler.load_state_dict(torch.load(path, map_location=device))
        print("Loaded scheduler from checkpoint.")
    else:
        print("Scheduler checkpoint file not found.")


def save_optimizer(optimizer: Optimizer, path: str):
    """
    Saves an optimizer to a path.
    :param optimizer: (torch.optim.Optimizer) The optimizer to save.
    :param path: (str) The path to save the optimizer to.
    """
    torch.save(optimizer.state_dict(), path)


def save_scheduler(scheduler: LRScheduler, path: str):
    torch.save(scheduler.state_dict(), path)


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: Optional[List[int]] = None, encounters=1):
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
