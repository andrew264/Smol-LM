import os
import time
from typing import Optional

import bitsandbytes as bnb
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from model import ModelConfig, LoRAConfig
from utils import load_model, load_optimizer, save_model, save_optimizer, count_parameters, load_scheduler, \
    save_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NPDataset(Dataset):
    def __init__(self, _path, block_size=1024, validation_split=False):
        validation_size = 5000 * block_size
        self.data = np.memmap(_path, dtype=np.uint16, mode='r')
        if validation_split:
            self.data = self.data[-validation_size:]
        else:
            self.data = self.data[:-validation_size]
        self.num_samples = len(self.data) // block_size
        self.data = self.data[:self.num_samples * block_size].reshape(self.num_samples, block_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> [np.ndarray]:
        return [np.int64(self.data[index])]


class TextCorpus(Dataset):
    """Please use smaller files; this is not memory efficient."""

    def __init__(self, _path: str, tokenizer: Tokenizer, block_size=1024):
        with open(_path, 'r') as _f:
            text = _f.read()
        self.data = np.array(tokenizer.encode(text).ids, dtype=np.uint16)
        del text
        self.num_samples = len(self.data) // block_size
        self.data = self.data[:self.num_samples * block_size].reshape(self.num_samples, block_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> [np.ndarray]:
        return [np.int64(self.data[index])]


@torch.no_grad()
def validate_model(model: Optional[nn.Module], validation_data: DataLoader, full_validation: bool = False):
    if not model:
        model = load_model(config=ModelConfig.from_json('./weights/config.json'),
                           path='weights/model.safetensors',
                           device=device)
        torch.compile(model=model.forward, fullgraph=True, mode='max-autotune')
    model.eval()
    total = len(validation_data) if full_validation or len(validation_data) < 100 else 100
    accumulated_loss = 0

    for (i, item) in tqdm.tqdm(enumerate(validation_data), total=total, desc="Validating"):
        ids = item[0].to(device)
        if len(item) == 3:
            labels, mask = item[1].to(device), item[2].to(device)
        else:
            labels, mask = ids, None
        with torch.no_grad():
            out = model(input_ids=ids, labels=labels, attention_mask=mask)
            logits, loss = out.logits, out.loss
            accumulated_loss += loss.item()

        if not full_validation and i > 99:
            break

    avg_loss = accumulated_loss / total
    avg_perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Validation | Loss: {avg_loss:.3f} | Perplexity: {avg_perplexity:.3f}")
    model.train()


def train(model_path: str, training_data: DataLoader, config: ModelConfig, lora_config: LoRAConfig = None,
          validation_data: Optional[DataLoader] = None, start_step: int = 0,
          save_step_count: bool = False, disable_scheduler: bool = False, learning_rate: float = 3e-4,
          save_every: int = 2000):
    """

    :param model_path: Model path to save model weights
    :param training_data: DataLoader for training data
    :param config: ModelConfig
    :param lora_config: LoRAConfig
    :param validation_data: DataLoader for validation data
    :param start_step: Start training from this step (useful for resuming training)
    :param save_step_count: Save the current step count to model_path/step.txt
    :param disable_scheduler: Disable the learning rate scheduler
    :param learning_rate: Learning rate for the optimizer
    :param save_every: Save the model weights every `save_every` steps
    :return:
    """

    model = load_model(config, lora_config=lora_config, path=model_path + 'model.safetensors', device=device)
    model.train()

    # accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config.grad_accumulation_steps)
    torch.compile(model=model.forward, fullgraph=True, mode='max-autotune')
    model = accelerator.prepare_model(model)

    # total steps
    total_steps = len(training_data) * config.epochs if isinstance(training_data, DataLoader) else None
    if total_steps is None:
        print("Could not determine total steps. Disabling scheduler.")
        disable_scheduler = True

    # optimizer
    betas = (0.9, 0.95)
    weight_decay = 0.1
    _params = model.get_optimizer_grouped_parameters(weight_decay)
    optimizer = bnb.optim.PagedAdamW8bit(_params, lr=learning_rate, betas=betas, weight_decay=weight_decay)
    optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)
    # TODO: figure out why loading optimizer states is using more memory
    # TODO: figure out why setting device to CPU or GPU use different amount of memory [GPU uses more]
    optimizer = load_optimizer(optimizer, model_path + 'optimizer.bin', device=device)
    optimizer.optimizer.to_gpu()

    # scheduler
    scheduler = None
    if not disable_scheduler:
        assert total_steps is not None
        s_steps = total_steps // config.grad_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(s_steps * 0.02),
                                                    num_training_steps=s_steps)
        scheduler = accelerator.prepare_scheduler(scheduler)
        scheduler = load_scheduler(scheduler, model_path + 'scheduler.bin', device=device)

    count_parameters(model)

    accumulated_loss = 0
    start_time, time_delta = time.time(), 0.
    print_step = max(1000, config.grad_accumulation_steps)
    model.train()
    print(
        f"Tokens per batch: {config.max_batch_size * config.grad_accumulation_steps * config.max_position_embeddings}")

    print(f"Training Step: {start_step} of {total_steps} | {start_step / total_steps * 100:.2f}%")

    for epoch in range(config.epochs):
        print(f"Epoch: {epoch + 1} of {config.epochs}")
        for (i, item) in enumerate(training_data):
            if i <= start_step:
                continue
            if i == start_step + 1:
                start_time = time.time()

            ids = item[0].to(device)
            if len(item) == 3:
                labels, mask = item[1].to(device), item[2].to(device)
            else:
                labels, mask = ids, None

            # train step
            with accelerator.accumulate(model):
                loss = model(input_ids=ids, labels=labels, attention_mask=mask).loss  # forward pass
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
                batch_per_sec = print_step / time_delta

                accelerator.print(f"Step: {i} | Loss: {avg_loss:.3f} | Perplexity: {avg_perplexity:.3f} | "
                                  f"Elapsed Time: {time_delta:.1f}s | Batch/sec: {batch_per_sec:.1f}")

                start_time = time.time()
                accumulated_loss = 0

            if i % save_every == 0 and i > 0:
                save_model(model, model_path + 'model.safetensors')
                save_optimizer(optimizer, model_path + 'optimizer.bin')
                if scheduler is not None:
                    save_scheduler(scheduler, model_path + 'scheduler.bin')
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

        if validation_data is not None:
            validate_model(model, validation_data, full_validation=True)

    save_model(model, model_path + 'model.safetensors')
    save_optimizer(optimizer, model_path + 'optimizer.bin')
    if scheduler is not None:
        save_scheduler(scheduler, model_path + 'scheduler.bin')

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
    training = NPDataset('./data/processed/train.bin', params.max_position_embeddings)
    validation = NPDataset('./data/processed/train.bin', params.max_position_embeddings, validation_split=True)
    # tokenizer = Tokenizer.from_file('./weights/tokenizer.json')
    # training = TextCorpus('./data/processed/train.txt', tokenizer, params.max_position_embeddings)
    # validation = training

    # resume training
    step = 0
    if os.path.exists(path + 'step.txt'):
        with open(path + 'step.txt', 'r') as f:
            step = int(f.read())

    train_data = DataLoader(training, batch_size=params.max_batch_size, shuffle=False, drop_last=True, )
    val_data = DataLoader(validation, batch_size=params.max_batch_size, shuffle=False, drop_last=True,
                          pin_memory=True)
    print("Train: ", len(train_data), "Validation: ", len(val_data))

    train(path,
          training_data=train_data,
          validation_data=val_data,
          config=params,
          start_step=step,
          save_step_count=True,
          disable_scheduler=False,
          learning_rate=5e-4
          )
    # validate_model(None, val_data, True)
