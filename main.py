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

    def __getitem__(self, index) -> (np.ndarray, np.ndarray):
        return np.int64(self.data[index][:-1]), np.int64(self.data[index][1:])


@torch.no_grad()
def estimate_loss(model: Transformer, config: ModelConfig):
    model.eval()
    dataset = NPDataset('./data/processed/val.bin', config.max_position_embeddings)
    validation_data = DataLoader(dataset, batch_size=config.max_batch_size, shuffle=True, drop_last=True)
    losses = []
    for i, (x, y) in enumerate(validation_data):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        losses.append(loss.item())
        if i >= 250:
            break
    avg_loss = sum(losses) / len(losses)
    avg_perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Validation | Loss {avg_loss:.3f} | Perplexity {avg_perplexity:.3f} | "
          f"Bits/Token {avg_loss / np.log(2):.3f}")
    model.train()


def train(model, config: ModelConfig):
    # dataloader
    dataset = NPDataset(dataset_path, config.max_position_embeddings)
    train_data = DataLoader(dataset, batch_size=config.max_batch_size, shuffle=False, drop_last=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, fused=True)

    accum_steps = config.grad_accumulation_steps

    losses = []
    start_time = time.time()
    step = 0
    if os.path.exists('./weights/step.txt'):
        with open('./weights/step.txt', 'r') as f:
            step = int(f.read())
    print(f"Starting from step {step} / {len(train_data)}")
    model.train()
    for i, (x, y) in enumerate(train_data):
        if i <= step:
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
            if (i + 1) % accum_steps == 0 or i == len(train_data) - 1:
                optimizer.step()
                optimizer.zero_grad()

        losses.append(loss.item())
        if i % 100 == 0 and i > 0:
            avg_loss = sum(losses) / len(losses)
            avg_perplexity = torch.exp(torch.tensor(avg_loss))
            print(f"Step {i} | Loss {avg_loss:.3f} | Perplexity {avg_perplexity:.3f} | "
                  f"Bits/Token {avg_loss / np.log(2):.3f} | "
                  f"Time {time.time() - start_time:.1f}s")
            start_time = time.time()
            losses = []
        if i % 1000 == 0 and i > 0:
            torch.save(model.state_dict(), f"./weights/model_ckpt.pt")
            with open('./weights/step.txt', 'w') as f:
                f.write(f"{i}\n")
            estimate_loss(model, config=config)
    torch.save(model.state_dict(), f"./weights/model_ckpt.pt")


if __name__ == '__main__':

    if os.path.exists('./weights/config.json'):
        config = ModelConfig.from_json('./weights/config.json')
        print("Loaded config from file.")
    else:
        config = ModelConfig()
        config.grad_accumulation_steps = 8
        config.max_batch_size = 8
        print("Created new config.")
        config.to_json('./weights/config.json')


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)
    torch.compile(model=model.forward, fullgraph=True, mode='reduce-overhead')
    print(f"Model has {count_parameters(model) / 1024 / 1024:.2f}M parameters.")

    if os.path.exists('./weights/model_ckpt.pt'):
        model.load_state_dict(torch.load('./weights/model_ckpt.pt'))
        print("Loaded model from file.")
    else:
        print("Created new model.")

    train(model, config=config)
    # estimate_loss(model, config=config)
