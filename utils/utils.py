import os
import time
from typing import Optional, List, TypeVar

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file


def get_state_dict_from_safetensors(path: str | List[str],
                                    device: torch.device = torch.device('cpu')) -> Optional[dict]:
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
        if device.type != 'cpu':
            torch.cuda.synchronize(device)
        print(f"Loaded weights from {path} in {time.time() - start:.3f}s.")
    else:
        print("No weights found.")
    return state_dict if state_dict else None


def save_as_safetensors(state_dict: dict, path: str):
    start = time.time()
    safe_save_file(state_dict, path)
    print(f"Saved state_dict to {path} in {time.time() - start:.3f}s.")


def save_state_dict(state_dict: dict, path: str):
    start = time.time()
    torch.save(state_dict, path)
    print(f"Saved state_dict to {path} in {time.time() - start:.3f}s.")


def get_state_dict(path: str, device: torch.device) -> Optional[dict]:
    """
    Load the saved `state_dict` from the given path.
    """
    start = time.time()
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        print(f"Loaded state_dict from {path} in {time.time() - start:.3f}s.")
        return state_dict
    else:
        return None


# T is any type that inherits from torch.nn.Module
T = TypeVar('T', bound=torch.nn.Module)


def compile_model(model: T) -> None:
    start = time.time()
    model.forward = torch.compile(model=model.forward, fullgraph=True, mode='max-autotune')
    torch.cuda.synchronize()
    print(f"Compiled model in {time.time() - start:.3f}s.")


def count_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params != total_trainable_params:
        print(f'{total_trainable_params:,} trainable parameters.')
        print(f'Percentage of trainable parameters: {total_trainable_params / total_params * 100:.2f}%')
