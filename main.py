import os
import time

import bitsandbytes as bnb
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from torch import nn
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
def estimate_loss(model: nn.Module, config: ModelConfig):
    model.eval()
    dataset = NPDataset('./data/processed/val.bin', config.max_position_embeddings)
    validation_data = DataLoader(dataset, batch_size=config.max_batch_size, shuffle=True, drop_last=True)
    losses = []
    for i, (x, y) in enumerate(validation_data):
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x=x, y=y)
        losses.append(loss.item())
        if i >= 250:
            break
    avg_loss = sum(losses) / len(losses)
    avg_perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Validation | Loss {avg_loss:.3f} | Perplexity {avg_perplexity:.3f} | "
          f"Bits/Token {avg_loss / np.log(2):.3f}")
    model.train()


def train(model: nn.Module, optimizer, config: ModelConfig):
    # dataloader
    dataset = NPDataset(dataset_path, config.max_position_embeddings)
    train_data = DataLoader(dataset, batch_size=config.max_batch_size, shuffle=False, drop_last=True)

    step = 0
    if os.path.exists('./weights/step.txt'):
        with open('./weights/step.txt', 'r') as f:
            step = int(f.read())

    accum_steps = config.grad_accumulation_steps
    accelerator = Accelerator(gradient_accumulation_steps=accum_steps)

    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5000,
                                                             num_training_steps=len(train_data) * config.max_epochs)
    scheduler.last_epoch = step

    model, optimizer, train_data, scheduler = accelerator.prepare(model, optimizer, train_data, scheduler)

    losses = []
    start_time = time.time()
    print(f"Starting from step {step} / {len(train_data)}")
    model.train()
    for i, (x, y) in enumerate(train_data):
        if i <= step:
            continue
        if i == step + 1:
            start_time = time.time()

        # train step
        with accelerator.accumulate(model):
            logits, loss = model(x=x, y=y)  # forward pass
            losses.append(loss.item())
            accelerator.backward(loss)  # backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if i % 100 == 0 and i > 0:
            time_delta = time.time() - start_time
            avg_loss = sum(losses) / len(losses)
            avg_perplexity = torch.exp(torch.tensor(avg_loss))
            tokens_per_sec = 100 * config.max_batch_size * config.max_position_embeddings / time_delta
            print(f"Step {i} | Loss {avg_loss:.3f} | Perplexity {avg_perplexity:.3f} | "
                  f"Time {time_delta:.1f}s | "
                  f"Tokens/s {tokens_per_sec:.1f}")
            start_time = time.time()
            losses = []
        if i % 1000 == 0 and i > 0:
            torch.save(model.state_dict(), f"./weights/model_ckpt.pt")
            with open('./weights/step.txt', 'w') as f:
                f.write(f"{i}\n")
            estimate_loss(model, config=config)
            start_time = time.time()
    torch.save(model.state_dict(), f"./weights/model_ckpt.pt")


if __name__ == '__main__':

    if os.path.exists('./weights/config.json'):
        config = ModelConfig.from_json('./weights/config.json')
        print("Loaded config from file.")
    else:
        config = ModelConfig()
        config.vocab_size = 32000
        print("Created new config.")
        config.to_json('./weights/config.json')


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)
    torch.compile(model=model.forward, fullgraph=True, mode='reduce-overhead')
    print(f"Model has {count_parameters(model) / 1e6:.2f}M parameters.")
    print(f"Model is_moe: {config.is_moe}")

    if os.path.exists('./weights/model_ckpt.pt'):
        model.load_state_dict(torch.load('./weights/model_ckpt.pt'))
        print("Loaded model from file.")
    else:
        print("Created new model.")

    # optimizer
    lr = 3e-4
    betas = (0.9, 0.95)
    weight_decay = 0.1
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr, betas=betas,
                                         weight_decay=weight_decay, min_8bit_size=0)

    train(model, optimizer, config=config)
    # estimate_loss(model, config=config)
