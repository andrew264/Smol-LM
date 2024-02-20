import os
import time
from typing import Optional

import bitsandbytes as bnb
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from model import ModelConfig, Transformer
from utils import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NPDataset(Dataset):
    def __init__(self, path, block_size=1024):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.num_samples = len(self.data) // block_size
        self.data = self.data[:self.num_samples * block_size].reshape(self.num_samples, block_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> [np.ndarray]:
        return [np.int64(self.data[index])]


def count_parameters(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def validate_model(model: Optional[nn.Module], validation_data: DataLoader, full_validation: bool = False):
    if not model:
        model = load_model(config=ModelConfig.from_json('./weights/config.json'),
                           path='./weights/accelerator_states/model.safetensors',
                           device=device)
    model.eval()
    total = len(validation_data) if full_validation or len(validation_data) < 100 else 100
    accumulated_loss = 0

    for i, item in tqdm.tqdm(enumerate(validation_data), total=total, desc="Validating"):
        ids, mask = item[0].to(device), item[1].to(device) if len(item) > 1 else None
        with torch.no_grad():
            logits, loss = model(input_ids=ids, labels=ids, mask=mask)
            accumulated_loss += loss.item()

        if not full_validation and i > 99:
            break

    avg_loss = accumulated_loss / total
    avg_perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Validation | Loss: {avg_loss:.3f} | Perplexity: {avg_perplexity:.3f}")
    model.train()


def train(model_path: str, training_data: DataLoader, config: ModelConfig,
          validation_data: Optional[DataLoader] = None, start_step: int = 0, save_step_count: bool = False,
          disable_grads_for_embeddings: bool = False, disable_scheduler: bool = False, learning_rate: float = 3e-4,
          save_every: int = 2000):
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
    :param save_every: Save the model weights every `save_every` steps
    :return:
    """

    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)

    torch.compile(model=model.forward, fullgraph=True, mode='max-autotune')
    print(f"Model has {count_parameters(model) / 1e6:.2f}M parameters.")

    if disable_grads_for_embeddings:
        for param in model.tok_embeddings.parameters():
            param.requires_grad = False
        print("Disabled gradients for embedding layer.")

    # total steps
    total_steps = len(training_data) if isinstance(training_data, DataLoader) else None
    if total_steps is None:
        print("Could not determine total steps. Disabling scheduler.")
        disable_scheduler = True

    # optimizer
    betas = (0.9, 0.95)
    weight_decay = 0.1
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate, betas=betas,
                                    weight_decay=weight_decay)

    # scheduler
    scheduler = None
    if not disable_scheduler:
        assert total_steps is not None
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(
                                                        total_steps // config.grad_accumulation_steps * 0.02),
                                                    num_training_steps=total_steps // config.grad_accumulation_steps)

    # accelerator
    checkpoint = model_path + "accelerator_states"
    accelerator = Accelerator(gradient_accumulation_steps=config.grad_accumulation_steps, project_dir=model_path)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if os.path.exists(checkpoint):
        # on God this fixes OOM while loading scheduler states on a single GPU
        accelerator.load_state(checkpoint, map_location='on_device')
        print("Loaded accelerator state from file.")
    else:
        accelerator.save_state(output_dir=checkpoint)
        print("Saved accelerator state to file.")

    accumulated_loss = 0
    start_time = time.time()
    print_step = save_every // 10
    model.train()
    print(f"Simulated Batch Size: {config.max_batch_size * config.grad_accumulation_steps}")

    for epoch in range(config.max_epochs):
        print(f"Starting Epoch: {epoch + 1} of {config.max_epochs}")
        print(f"Training Step: {start_step} of {total_steps} | {start_step / total_steps * 100:.2f}%")

        for i, item in enumerate(training_data):
            if i <= start_step:
                continue
            if i == start_step + 1:
                start_time = time.time()

            ids, mask = item[0], item[1] if len(item) > 1 else None
            ids, mask = ids.to(device), mask.to(device) if mask is not None else None

            # train step
            with accelerator.accumulate(model):
                logits, loss = model(input_ids=ids, labels=ids, mask=mask)  # forward pass
                accumulated_loss += loss.item()
                accelerator.backward(loss)  # backward pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad()

            if i % print_step == 0 and i > 0:
                time_delta = time.time() - start_time
                avg_loss = accumulated_loss / print_step
                avg_perplexity = torch.exp(torch.tensor(avg_loss))
                samples_per_min = print_step * config.max_batch_size / (time_delta / 60)

                accelerator.print(f"Step: {i} | Loss: {avg_loss:.3f} | Perplexity: {avg_perplexity:.3f} | "
                                  f"Elapsed Time: {time_delta:.1f}s | Samples/Min: {samples_per_min:.1f}")

                start_time = time.time()
                accumulated_loss = 0

            if i % save_every == 0 and i > 0:
                accelerator.save_state(output_dir=checkpoint)

                if save_step_count:
                    with open(model_path + 'step.txt', 'w') as step_file:
                        step_file.write(f"{i}\n")

                if validation_data is not None:
                    if i % 10000 == 0:
                        validate_model(model, validation_data, full_validation=True)
                    else:
                        validate_model(model, validation_data)

                start_time = time.time()

        accelerator.save_state(output_dir=checkpoint)
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
    train_data = DataLoader(dataset, batch_size=params.max_batch_size, shuffle=False, drop_last=True,
                            pin_memory=True)
    # validation
    dataset = NPDataset('./data/processed/val.bin', params.max_position_embeddings)
    val_data = DataLoader(dataset, batch_size=params.max_batch_size, shuffle=False, drop_last=True, pin_memory=True)

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
          disable_grads_for_embeddings=False,
          learning_rate=3e-4
          )
    # validate_model(None, val_data, True)
