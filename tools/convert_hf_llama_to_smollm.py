import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file


def get_updated_key(k: str) -> str:
    return (k
            .replace("model.", "")
            .replace("embed_tokens", "tok_embeddings")
            .replace("self_attn", "attention")
            .replace("mlp", "feed_forward"))


if __name__ == '__main__':
    input_path = '/mnt/d/LLMs/TinyLlama-1.1B-Chat-v1.0/model.safetensors'
    assert os.path.exists(input_path), f"Path {input_path} does not exist."
    output_path = '../weights/model.safetensors'
    state_dict = {}
    device = torch.device("cpu")
    d = device.type if device.type == 'cpu' else device.index
    print("Converting weights from Llama to somllm...")
    with safe_open(input_path, framework="pt", device=d) as f:
        for k in f.keys():
            state_dict[get_updated_key(k)] = f.get_tensor(k)

    safe_save_file(state_dict, output_path)
    print("Converted weights from Llama to somllm.")
