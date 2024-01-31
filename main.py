import os
import time
from typing import Optional, Tuple

import bitsandbytes as bnb
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from model import ModelConfig
from utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NPDataset(Dataset):
    def __init__(self, path, block_size=1024):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.num_samples = len(self.data) // block_size
        self.data = self.data[:self.num_samples * block_size].reshape(self.num_samples, block_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        ids = np.int64(self.data[index])
        return ids, ids


def count_parameters(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def validate_model(model: Optional[nn.Module], validation_data: DataLoader, full_validation: bool = False):
    if not model:
        model = load_model(ModelConfig.from_json('./weights/config.json'), './weights/model_ckpt.pt', device)
    model.eval()

    losses = []
    for i, item in tqdm.tqdm(enumerate(validation_data),
                             total=len(validation_data) if full_validation or len(validation_data) < 100 else 100,
                             desc="Validating"):
        ids, labels, mask = item[0].to(device), item[1].to(device), item[2].to(device) if len(item) > 2 else None
        with torch.no_grad():
            logits, loss = model(input_ids=ids, labels=labels, mask=mask)
            losses.append(loss.item())

        if not full_validation and i > 99:
            break

    avg_loss = sum(losses) / len(losses)
    avg_perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Validation | Loss: {avg_loss:.3f} | Perplexity: {avg_perplexity:.3f}")
    model.train()


def train(model_path: str, training_data: DataLoader, config: ModelConfig,
          validation_data: Optional[DataLoader] = None, start_step: int = 0, save_step_count: bool = False,
          disable_grads_for_embeddings: bool = False, disable_scheduler: bool = False, learning_rate: float = 3e-4):
    """

    :param model_path: Model path to save model weights
    :param training_data: DataLoader for training data
    :param config: ModelConfig
    :param validation_data: DataLoader for validation data
    :param start_step: Start training from this step (useful for resuming training)
    :param save_step_count: Save the current step count to model_path/step.txt
    :param disable_grads_for_embeddings: Disable gradients for embedding layer and the output layer
    :param disable_scheduler: Disable the learning rate scheduler
    :param learning_rate: Learning rate for the optimizer
    :return:
    """

    model_weights_path = model_path + "model_ckpt.pt"
    model = load_model(config, model_weights_path, device)

    torch.compile(model=model.forward, fullgraph=True, mode='max-autotune')
    print(f"Model has {count_parameters(model) / 1e6:.2f}M parameters.")

    if disable_grads_for_embeddings:
        for param in model.tok_embeddings.parameters():
            param.requires_grad = False
        for param in model.output.parameters():
            param.requires_grad = False
        print("Disabled gradients for embedding layer and output layer.")

    # total steps
    total_steps = len(training_data) if isinstance(training_data, DataLoader) else None
    if total_steps is None:
        print("Could not determine total steps. Disabling scheduler.")
        disable_scheduler = True

    # optimizer
    betas = (0.9, 0.95)
    weight_decay = 0.1
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=learning_rate, betas=betas,
                                         weight_decay=weight_decay, min_8bit_size=config.hidden_size, )

    # scheduler
    scheduler = None
    if not disable_scheduler:
        assert total_steps is not None
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.01),
                                                    num_training_steps=total_steps * config.max_epochs)
        scheduler.last_epoch = start_step - 1

    # accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config.grad_accumulation_steps)
    model, optimizer, data, scheduler = accelerator.prepare(model, optimizer, training_data, scheduler)

    losses = []
    start_time = time.time()
    print_step = 250
    model.train()

    for epoch in range(config.max_epochs):
        print(f"Starting Epoch: {epoch + 1} of {config.max_epochs}")
        print(f"Training Step: {start_step} of {total_steps}")

        for i, item in enumerate(data):
            if i <= start_step:
                continue
            if i == start_step + 1:
                start_time = time.time()

            ids, labels, mask = item[0], item[1], item[2].to(device) if len(item) > 2 else None

            # train step
            with accelerator.accumulate(model):
                logits, loss = model(input_ids=ids, labels=labels, mask=mask)  # forward pass
                losses.append(loss.item())
                accelerator.backward(loss)  # backward pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if not disable_scheduler:
                    scheduler.step()

                optimizer.zero_grad()

            if i % print_step == 0 and i > 0:
                time_delta = time.time() - start_time
                avg_loss = sum(losses) / len(losses)
                avg_perplexity = torch.exp(torch.tensor(avg_loss))
                tokens_per_sec = print_step * config.max_batch_size * config.max_position_embeddings / time_delta

                print(f"Step: {i} | Loss: {avg_loss:.3f} | Perplexity: {avg_perplexity:.3f} | "
                      f"Elapsed Time: {time_delta:.1f}s | Tokens per Second: {tokens_per_sec:.1f}")

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
          disable_scheduler=False,
          learning_rate=5e-5
          )
    # validate_model(None, val_data, True)
