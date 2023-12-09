import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import ModelConfig, Transformer

dataset_path = './data/processed/train.bin'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


class NPDataset(Dataset):
    def __init__(self, path, block_size=1024):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.num_samples = len(self.data) // (block_size + 1)
        self.data = np.reshape(self.data[:self.num_samples * (block_size + 1)], (-1, block_size + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return np.int64(self.data[index][:-1]), np.int64(self.data[index][1:])


def train(model, batch_size: int, config: ModelConfig):
    # dataloader
    dataset = NPDataset(dataset_path, config.max_position_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    accum_steps = config.grad_accumulation_steps

    losses = []
    start_time = time.time()
    step = 0
    if os.path.exists('./weights/step.txt'):
        with open('./weights/step.txt', 'r') as f:
            step = int(f.read())
    for i, (x, y) in enumerate(dataloader):
        if i < step:
            continue
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(True):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            losses.append(loss.item())
            loss = loss / accum_steps
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (i + 1) % accum_steps == 0 or i == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()

        losses.append(loss.item())
        if i % 100 == 0 and i > 0:
            avg_loss = sum(losses) / len(losses)
            avg_perplexity = 2 ** avg_loss
            elapsed = time.time() - start_time
            print(f"Step {i} | Loss {avg_loss:.3f} | Perplexity {avg_perplexity:.3f} | "
                  f"Bits/Token {avg_loss / np.log(2):.3f} | "
                  f"Tokens/s {config.max_position_embeddings * len(losses) * batch_size / elapsed:.0f}")
            losses = []
            start_time = time.time()
        if i % 10000 == 0 and i > 0:
            torch.save(model.state_dict(), f"./weights/model_ckpt.pt")
            with open('./weights/step.txt', 'w') as f:
                f.write(f"{i}\n")
        torch.save(model.state_dict(), f"./weights/model_ckpt.pt")


if __name__ == '__main__':
    batch = 12

    if os.path.exists('./weights/config.json'):
        config = ModelConfig.from_json('./weights/config.json')
        print("Loaded config from file.")
    else:
        config = ModelConfig()
        print("Created new config.")
        config.to_json('./weights/config.json')


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)
    model.setup_caches(max_batch_size=batch, max_seq_length=config.max_position_embeddings, device=device)
    # torch.compile(model=model.forward, fullgraph=True, mode='reduce-overhead')
    print(f"Model has {count_parameters(model) / 1024 / 1024:.2f}M parameters.")

    if os.path.exists('./weights/model_ckpt.pt'):
        model.load_state_dict(torch.load('./weights/model_ckpt.pt'))
        print("Loaded model from file.")
    else:
        print("Created new model.")

    train(model, batch_size=batch, config=config)
