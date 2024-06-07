import os
import time
from itertools import cycle
from typing import Optional, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig


def get_state_dict_from_safetensors(path: str | List[str], device: torch.device) -> Optional[dict]:
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


def compile_model(model: torch.nn.Module, ) -> None:
    start = time.time()
    torch.compile(model=model.forward, fullgraph=True, mode='max-autotune')
    torch.cuda.synchronize()
    print(f"Compiled model in {time.time() - start:.3f}s.")


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


def get_generation_config(max_new_length: int) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=max_new_length,
        do_sample=True,
        num_beams=1,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


class CyclingDataLoader:
    """
    Just a bunch of DataLoaders that cycle through each other.
    """

    def __init__(self, *iterators):
        self.iterators = [iter(it) for it in iterators]
        self.cycle = cycle(self.iterators)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iterators:
            raise StopIteration

        while True:
            current = next(self.cycle)
            try:
                return next(current)
            except StopIteration:
                print(f"Removing exhausted iterator: {current}")
                self.iterators.remove(current)
                if not self.iterators:
                    raise StopIteration
                self.cycle = cycle(self.iterators)
