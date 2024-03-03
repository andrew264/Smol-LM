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

from model import ModelConfig
from utils import load_model, load_optimizer, save_model, save_optimizer

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
                           path='weights/model.safetensors',
                           device=device)
    model.eval()
    total = len(validation_data) if full_validation or len(validation_data) < 100 else 100
    accumulated_loss = 0

    for i, item in tqdm.tqdm(enumerate(validation_data), total=total, desc="Validating"):
        ids, mask = item[0].to(device), item[1].to(device) if len(item) > 1 else None
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids, attention_mask=mask)
            logits, loss = out.logits, out.loss
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

    model = load_model(config, model_path + 'model.safetensors', device=device)
    model.train()

    if disable_grads_for_embeddings:
        for param in model.tok_embeddings.parameters():
            param.requires_grad = False
        print("Disabled gradients for embedding layer.")

    # accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config.grad_accumulation_steps)
    model = accelerator.prepare_model(model)

    # total steps
    total_steps = len(training_data) if isinstance(training_data, DataLoader) else None
    if total_steps is None:
        print("Could not determine total steps. Disabling scheduler.")
        disable_scheduler = True

    # optimizer
    betas = (0.9, 0.95)
    weight_decay = 0.1
    optimizer = bnb.optim.AdamW8bit(model.get_optimizer_grouped_parameters(weight_decay),
                                    lr=learning_rate, betas=betas, )
    optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)
    # TODO: figure out why loading optimizer states is using more memory
    # TODO: figure out why setting device to CPU or GPU use different amount of memory [GPU uses more]
    optimizer = load_optimizer(optimizer, model_path + 'optimizer.bin', device=device)
    optimizer.optimizer.to_gpu()

    # scheduler
    scheduler = None
    if not disable_scheduler:
        assert total_steps is not None
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(
                                                        total_steps // config.grad_accumulation_steps * 0.02),
                                                    num_training_steps=total_steps // config.grad_accumulation_steps)

    torch.compile(model=model.forward, fullgraph=True, mode='max-autotune')
    print(f"Model has {count_parameters(model) / 1e6:.2f}M parameters.")

    accumulated_loss = 0
    start_time, time_delta = time.time(), 0.
    print_step = max(save_every // 10, config.grad_accumulation_steps)
    model.train()
    print(
        f"Tokens per batch: {config.max_batch_size * config.grad_accumulation_steps * config.max_position_embeddings}")

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
            out = model(input_ids=ids, labels=ids, attention_mask=mask)  # forward pass
            logits, loss = out.logits, out.loss
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
            tokens_per_sec = print_step * config.max_batch_size * config.max_position_embeddings / time_delta

            accelerator.print(f"Step: {i} | Loss: {avg_loss:.3f} | Perplexity: {avg_perplexity:.3f} | "
                              f"Elapsed Time: {time_delta:.1f}s | Tokens/sec: {tokens_per_sec:.0f}")

            start_time = time.time()
            accumulated_loss = 0

        if i % save_every == 0 and i > 0:
            save_model(model, model_path + 'model.safetensors')
            save_optimizer(optimizer, model_path + 'optimizer.bin')
            print(f"Percent of dataset consumed: {i / total_steps * 100:.2f}% | "
                  f"Time left: {((total_steps - i) * (time_delta / print_step)) / 60:.2f} minutes")

            if save_step_count:
                with open(model_path + 'step.txt', 'w') as step_file:
                    step_file.write(f"{i}\n")

            if validation_data is not None:
                if i % 10000 == 0:
                    validate_model(model, validation_data, full_validation=True)
                else:
                    validate_model(model, validation_data)

            start_time = time.time()

    save_model(model, model_path + 'model.safetensors')
    save_optimizer(optimizer, model_path + 'optimizer.bin')

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
        print("Created new config.")
        params.to_json(path + 'config.json')

    # training
    dataset = NPDataset('./data/processed/train.bin', params.max_position_embeddings)
    gen = torch.Generator().manual_seed(42)
    training, validation = torch.utils.data.random_split(dataset,
                                                         [int(len(dataset) * 0.9995),
                                                          len(dataset) - int(len(dataset) * 0.9995)],
                                                         generator=gen)

    # resume training
    step = 0
    if os.path.exists(path + 'step.txt'):
        with open(path + 'step.txt', 'r') as f:
            step = int(f.read())

    train_data = DataLoader(training, batch_size=params.max_batch_size, shuffle=False, drop_last=True, num_workers=20, )
    val_data = DataLoader(validation, batch_size=params.max_batch_size, shuffle=False, drop_last=True,
                          pin_memory=True)
    print("Train: ", len(train_data), "Validation: ", len(val_data))

    train(path,
          training_data=train_data,
          validation_data=val_data,
          config=params,
          start_step=step,
          save_step_count=True,
          disable_scheduler=True,
          disable_grads_for_embeddings=False,
          learning_rate=5e-4
          )
    # validate_model(None, val_data, True)
