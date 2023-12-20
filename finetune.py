import glob
import os
import pickle
import time

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from model import ModelConfig, Transformer

dataset_paths = glob.glob('./data/processed-finetune/*.pkl', recursive=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PickleDataset(Dataset):
    def __init__(self, path, block_size=1024):
        self.data = pickle.load(open(path, 'rb'))
        # pad 0 to the end of each sequence
        self.data = [x + [0] * ((block_size + 1) - len(x)) for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> (np.ndarray, np.ndarray):
        return np.int64(self.data[index][:-1]), np.int64(self.data[index][1:])


def train(model, config: ModelConfig):
    epochs = config.max_epochs

    train_data = torch.utils.data.ConcatDataset([
        PickleDataset(path, config.max_position_embeddings) for path in dataset_paths
    ])
    dataloader = DataLoader(train_data, batch_size=config.max_batch_size, shuffle=True, drop_last=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, fused=True)

    accum_steps = config.grad_accumulation_steps
    accelerator = Accelerator(gradient_accumulation_steps=accum_steps)
    model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, train_data)

    losses = []
    start_time = time.time()
    print(f"Training for {epochs} epochs...")
    print(f"Total number of batches: {len(dataloader)}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} started.")
        for i, (x, y) in enumerate(dataloader):
            # train step
            with accelerator.accumulate(model):
                x = x.to(device)
                y = y.to(device)
                logits, loss = model(x=x, y=y)
                losses.append(loss.item())
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                time_delta = time.time() - start_time
                avg_loss = sum(losses) / len(losses)
                avg_perplexity = torch.exp(torch.tensor(avg_loss))
                tokens_per_sec = 100 * config.max_batch_size * config.max_position_embeddings / time_delta
                print(f"Step {i} | Loss {avg_loss:.3f} | Perplexity {avg_perplexity:.3f} | "
                      f"Bits/Token {avg_loss / np.log(2):.3f} | "
                      f"Time {time_delta:.1f}s | "
                      f"Tokens/s {tokens_per_sec:.1f}"
                      )
                start_time = time.time()
                losses = []
            if i % 1000 == 0 and i > 0:
                torch.save(model.state_dict(), f"./finetuned-weights/model_ckpt.pt")
        print(f"Epoch {epoch + 1} finished.")
        torch.save(model.state_dict(), f"./finetuned-weights/model_ckpt.pt")
    print("Training finished.")


if __name__ == '__main__':
    if os.path.exists('./weights/config.json'):
        config = ModelConfig.from_json('./weights/config.json')
        config.max_epochs = 2
        print("Loaded config from file.")
    else:
        raise FileNotFoundError("No config file found.")

    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)
    torch.compile(model=model.forward, fullgraph=True, mode='reduce-overhead')

    if os.path.exists('./finetuned-weights/model_ckpt.pt'):
        model.load_state_dict(torch.load('./finetuned-weights/model_ckpt.pt'))
        print("Continuing training from file.")
    elif os.path.exists('./weights/model_ckpt.pt'):
        model.load_state_dict(torch.load('./weights/model_ckpt.pt'))
        print("Loaded model from file.")
    else:
        raise FileNotFoundError("No model file found.")

    train(model, config)
