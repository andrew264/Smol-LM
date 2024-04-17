import os
import time
from typing import Optional, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig

from model import ModelConfig, SmolLM, LoRAConfig
from utils.lora_utils import to_lora_model


def load_model(config: ModelConfig,
               lora_config: Optional[LoRAConfig] = None,
               path: str | List[str] = None,
               device: torch.device = torch.device('cuda'),
               dtype: torch.dtype = torch.bfloat16
               ) -> SmolLM:
    state_dict = {}
    if isinstance(path, str):
        path = [path]
    if path and os.path.exists(path[0]):
        start = time.time()
        d = device.type if device.type == 'cpu' else device.index
        for p in path:
            with safe_open(p, framework="pt", device=d) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        print(f"Loaded weights from {path} in {time.time() - start:.3f}s.")
    else:
        print("Created new model.")

    start = time.time()
    is_lora_state = any('lora' in k for k in state_dict.keys())
    model = SmolLM(config).to(device=device, dtype=dtype)
    if is_lora_state:
        assert lora_config is not None, "LoRA config must be provided if model weights have LoRA."
        model = to_lora_model(model, lora_config)
        if state_dict:
            model.load_state_dict(state_dict)
    else:
        if state_dict:
            model.load_state_dict(state_dict)
        if lora_config is not None:
            model = to_lora_model(model, lora_config)
    del state_dict
    torch.cuda.empty_cache()
    model.eval()
    print(f"Loaded model in {time.time() - start:.3f}s.")
    return model


def save_model(model: torch.nn.Module, path: str):
    """
    Saves a model to a path as a .safetensors file.
    :param model: (torch.nn.Module) The model to save.
    :param path: (str) The path to save the model to.
    """
    start = time.time()
    safe_save_file(model.state_dict(), path)
    print(f"Saved model to {path} in {time.time() - start:.3f}s.")


def compile_model(model: torch.nn.Module, ) -> None:
    """
    Compiles a model to optimize performance.
    :param model: (torch.nn.Module) The model to compile.
    :return: (torch.nn.Module) The compiled model.
    """
    start = time.time()
    torch.compile(model=model.forward, fullgraph=True, mode='max-autotune')
    print(f"Compiled model in {time.time() - start:.3f}s.")


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

    def __init__(self, stops: Optional[List[torch.Tensor]] = None, encounters=1):
        super().__init__()
        if stops is None:
            stops = []
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        stop_count = 0
        for batch in input_ids:
            for stop in self.stops:
                if torch.equal(stop, batch[-len(stop):]):
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


def get_stopping_criteria(device: torch.device) -> StoppingCriteriaList:
    stopping_tokens: List[torch.Tensor] = [torch.tensor([i], device=device) for i in range(3)]
    stopping_tokens.append(torch.tensor([523, 28766], device=device))
    stopping_tokens.append(torch.tensor([28789, 28766], device=device))
    return StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])


def get_generation_config(max_length: int) -> GenerationConfig:
    return GenerationConfig(
        max_length=max_length,
        do_sample=True,
        num_beams=1,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
