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
    dataset = NPDataset(dataset_path, config.block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    losses = []
    perplexities = []
    start_time = time.time()
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        optimizer.zero_grad()  # reset gradients
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        perplexities.append(2 ** loss.item())
        if i % 100 == 0:
            avg_loss = sum(losses) / len(losses)
            avg_perplexity = sum(perplexities) / len(perplexities)
            elapsed = time.time() - start_time
            print(f"Step {i} | Loss {avg_loss:.3f} | Perplexity {avg_perplexity:.3f} | "
                  f"Bits/Token {avg_loss / np.log(2):.3f} | "
                  f"Tokens/s {config.block_size * len(losses) * batch_size / elapsed:.0f}")
            losses = []
            start_time = time.time()
        if i % 10000 == 0:
            torch.save(model.state_dict(), f"./weights/model_ckpt.pt")


if __name__ == '__main__':
    batch = 4

    if os.path.exists('./weights/config.json'):
        config = ModelConfig.from_json('./weights/config.json')
        print("Loaded config from file.")
    else:
        config = ModelConfig()
        print("Created new config.")
        config.to_json('./weights/config.json')

    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)
    model.setup_caches(max_batch_size=batch, max_seq_length=config.block_size, device=device)
    # torch.compile(model=model.forward, fullgraph=True, mode='reduce-overhead')

    if os.path.exists('./weights/model_ckpt.pt'):
        model.load_state_dict(torch.load('./weights/model_ckpt.pt'))
        print("Loaded model from file.")
    else:
        print("Created new model.")

    train(model, batch_size=batch, config=config)
