import re
from typing import Optional

import torch

from model import Tokenizer, ModelConfig, Transformer

weights = './weights/model_ckpt.pt'
tokenizer_path = './weights/tokenizer.model'
config = './weights/config.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def multinomial_sample_one_no_sync(probs_sort):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.no_grad()
def generate(
        model: Transformer,
        prompt: torch.Tensor,
        max_tokens: int,
        **sampling_kwargs
):
    while len(prompt) < max_tokens:
        logits = model(prompt.view(1, -1))
        idx_next, _ = sample(logits, **sampling_kwargs)
        out = tokenizer.decode_piece(idx_next.tolist()[0])
        out = out.replace('▁', ' ')
        if match := re.match(r'<0x([0-9a-fA-F]+)>', out):
            out = bytes.fromhex(match.group(1)).decode('utf-8', errors='ignore')
        print(out, end='', flush=True)
        prompt = torch.cat([prompt, idx_next], dim=-1)
    print()


if __name__ == '__main__':

    config = ModelConfig.from_json(config)
    config.max_batch_size = 1

    tokenizer = Tokenizer(tokenizer_path)
    model = Transformer(config)
    checkpoint = torch.load(weights, mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True, strict=False)
    model.to(dtype=torch.bfloat16, device=device)
    model = model.eval()

    print('model loaded')
    while True:
        prompt = input("Enter a prompt: ")
        if prompt == '':
            break
        tokens = tokenizer.encode(prompt)
        generate(model, torch.tensor(tokens, device=device, dtype=torch.int),
                 max_tokens=350, top_k=8)
