import os
import time
from typing import Optional

import bitsandbytes as bnb
import numpy as np
import torch
import tqdm
import transformers
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import ModelConfig, Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NPDataset(Dataset):
    def __init__(self, path, block_size=1024):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.num_samples = len(self.data) // (block_size + 1)
        self.data = np.reshape(self.data[:self.num_samples * (block_size + 1)], (-1, block_size + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> (np.ndarray, np.ndarray):
        x, y = np.int64(self.data[index][:-1]), np.int64(self.data[index][1:])
        return x, y


def count_parameters(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def validate_model(model: nn.Module, validation_data: DataLoader, full_validation: bool = False):
    model.eval()

    losses = []
    start_time = time.time()
    for i, (x, y) in tqdm.tqdm(enumerate(validation_data), total=len(validation_data) if full_validation else 100,
                               desc="Validating"):
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x=x, y=y)
        losses.append(loss.item())

        if not full_validation and i > 99:
            break

    avg_loss = sum(losses) / len(losses)
    avg_perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Validation | Loss {avg_loss:.3f} | Perplexity {avg_perplexity:.3f}"
          f" | Time {time.time() - start_time:.1f}s")
    model.train()


def train(model_path: str, training_data: DataLoader, config: ModelConfig,
          validation_data: Optional[DataLoader] = None, start_step: int = 0, save_step_count: bool = False,
          disable_grads_for_embeddings: bool = False):
    """

    :param model_path: Model path to save model weights
    :param training_data: DataLoader for training data
    :param config: ModelConfig
    :param validation_data: DataLoader for validation data
    :param start_step: Start training from this step (useful for resuming training)
    :param save_step_count: Save the current step count to model_path/step.txt
    :param disable_grads_for_embeddings: Disable gradients for embedding layer and the output layer
    :return:
    """

    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)
    torch.compile(model=model.forward, fullgraph=True, mode='reduce-overhead')
    print(f"Model has {count_parameters(model) / 1e6:.2f}M parameters.")

    model_weights_path = model_path + "model_ckpt.pt"
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path))
        print("Loaded model from weights file.")
    else:
        print("Created new model.")

    if disable_grads_for_embeddings:
        for param in model.tok_embeddings.parameters():
            param.requires_grad = False
        for param in model.output.parameters():
            param.requires_grad = False
        print("Disabled gradients for embedding layer and output layer.")

    # total steps
    try:
        total_steps = len(training_data)
    except TypeError:
        total_steps = int(1e6)

    # optimizer
    lr = 3e-4
    betas = (0.9, 0.95)
    weight_decay = 0.1
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr, betas=betas,
                                         weight_decay=weight_decay, min_8bit_size=config.hidden_size,)

    # scheduler
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5000,
                                                             num_training_steps=total_steps * config.max_epochs)
    scheduler.last_epoch = start_step

    # accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config.grad_accumulation_steps)
    model, optimizer, data, scheduler = accelerator.prepare(model, optimizer, training_data, scheduler)

    losses = []
    start_time = time.time()
    model.train()
    for epoch in range(config.max_epochs):
        print(f"Epoch {epoch + 1} / {config.max_epochs}")
        print(f"Starting from step {start_step} / {total_steps}")
        for i, (x, y) in enumerate(data):
            if i <= start_step:
                continue
            if i == start_step + 1:
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
                torch.save(model.state_dict(), model_weights_path)
                if save_step_count:
                    with open(model_path + 'step.txt', 'w') as step_file:
                        step_file.write(f"{i}\n")
                if validation_data is not None:
                    if i % 10000 == 0:
                        validate_model(model, validation_data, full_validation=True)
                    else:
                        validate_model(model, validation_data)
                start_time = time.time()
        torch.save(model.state_dict(), model_weights_path)
        start_step = 0
        if validation_data is not None:
            validate_model(model, validation_data, full_validation=True)
    print("Training complete.")


if __name__ == '__main__':

    path = './weights/'

    if os.path.exists(path + 'config.json'):
        params = ModelConfig.from_json(path + 'config.json')
        print("Loaded config from file.")
    else:
        params = ModelConfig()
        params.vocab_size = 32000
        print("Created new config.")
        params.to_json(path + 'config.json')

    # training
    dataset = NPDataset('./data/processed/train.bin', params.max_position_embeddings)
    train_data = DataLoader(dataset, batch_size=params.max_batch_size, shuffle=False, drop_last=True)
    # validation
    dataset = NPDataset('./data/processed/val.bin', params.max_position_embeddings)
    val_data = DataLoader(dataset, batch_size=params.max_batch_size, shuffle=False, drop_last=True)

    # resume training
    step = 0
    if os.path.exists(path + 'step.txt'):
        with open(path + 'step.txt', 'r') as f:
            step = int(f.read())

    train(path,
          training_data=train_data,
          validation_data=val_data,
          config=params,
          start_step=step,
          save_step_count=True,
          )
