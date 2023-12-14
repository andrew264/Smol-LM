import glob
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
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

    losses = []
    start_time = time.time()
    print(f"Training for {epochs} epochs...")
    print(f"Total number of batches: {len(dataloader)}")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} started.")
        for i, (x, y) in enumerate(dataloader):
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

            if (i + 1) % 100 == 0:
                avg_loss = sum(losses) / len(losses)
                avg_perplexity = torch.exp(torch.tensor(avg_loss))
                print(f"Epoch {epoch + 1} | Batch {i + 1} | Loss {avg_loss:.3f} | Perplexity {avg_perplexity:.3f} | "
                      f"Bits/Token {avg_loss / np.log(2):.3f} | "
                      f"Time {time.time() - start_time:.1f}s")
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
        config.max_epochs = 3
        print("Loaded config from file.")
    else:
        raise FileNotFoundError("No config file found.")

    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)
    torch.compile(model=model.forward, fullgraph=True, mode='reduce-overhead')

    if os.path.exists('./weights/model_ckpt.pt'):
        model.load_state_dict(torch.load('./weights/model_ckpt.pt'))
        print("Loaded model from file.")
    else:
        raise FileNotFoundError("No model file found.")

    train(model, config)
