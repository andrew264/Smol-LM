import datetime
import os
import time
from typing import Tuple, List

import bitsandbytes as bnb
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from main import validate_model  # noqa
from model import ModelConfig, LoRAConfig, SmolLM
from utils import (DiscordConversations,
                   get_state_dict_from_safetensors, compile_model,
                   inject_lora_adapter, get_lora_state_dict, get_lora_plus_optimizer_group,
                   save_as_safetensors, count_parameters)  # noqa

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CROSS_ENTROPY_IGNORE_IDX = -100


def move_to_device(batch):
    return {k: v.to(device) for k, v in batch.items()}


def train(_path: str,
          training_data: DataLoader,
          validation_data: DataLoader,
          config: ModelConfig,
          lora_config: LoRAConfig,
          disable_scheduler: bool = False,
          learning_rate: float = 2e-5,
          ):
    # Load model
    model_sd = get_state_dict_from_safetensors(os.path.join(_path, 'model.safetensors'), device)
    model = SmolLM(config).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(model_sd)
    del model_sd

    # Inject LoRA
    model = inject_lora_adapter(model, lora_config, )

    count_parameters(model)
    total_steps = len(training_data) * config.epochs
    print(f"Total steps: {total_steps} | Steps per epoch: {len(training_data)}")

    # Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config.grad_accumulation_steps)
    compile_model(model)
    model = accelerator.prepare_model(model)

    # Optimizer
    betas = (0.9, 0.999)
    optimizer = bnb.optim.AdamW8bit(get_lora_plus_optimizer_group(model, lr=learning_rate), betas=betas)
    optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)

    # Scheduler
    scheduler = None
    if not disable_scheduler:
        s_steps = total_steps // config.grad_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=10,
                                                    num_training_steps=s_steps)
        scheduler = accelerator.prepare_scheduler(scheduler)

    # Training loop
    torch.cuda.empty_cache()
    print("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        accu_loss = 0
        start = time.time()
        # Start of epoch
        for i, (item) in enumerate(training_data):
            with accelerator.accumulate(model):
                loss = model(**move_to_device(item)).loss
                accu_loss += loss.item()
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                if not disable_scheduler:
                    scheduler.step()
                optimizer.zero_grad()

        # End of epoch
        avg_loss = accu_loss / len(training_data)
        avg_ppl = torch.exp(torch.tensor(avg_loss))
        lrs = [f"{param_group['lr']:.1e}" for param_group in optimizer.param_groups]
        print(f"Epoch {epoch + 1} took {time.time() - start:.1f}s | "
              f"Loss: {avg_loss:.3f} | Perplexity: {avg_ppl:.3f} | LR: {lrs}")
        validate_model(model, validation_data, full_validation=True)

        # Save model
        adapter_sd = get_lora_state_dict(model)
        save_as_safetensors(adapter_sd, os.path.join(_path, 'adapter.safetensors'))


if __name__ == '__main__':
    from tokenizers import Tokenizer

    tokenizer_path = 'weights/tokenizer.json'
    tokenizer = Tokenizer.from_file(path=tokenizer_path)

    path = './ft-weights/'

    if os.path.exists(path + 'config.json'):
        params = ModelConfig.from_json(path + 'config.json')
        print("Loaded config from file.")
    else:
        raise ValueError("Config not found.")

    lora_params = None
    if os.path.exists(path + 'lora.json'):
        lora_params = LoRAConfig.from_json(path + 'lora.json')
        print("Loaded LoRA config from file.")
    else:
        lora_params = LoRAConfig()
        print("Created new LoRA config.")
        lora_params.to_json(path + 'lora.json')

    dt = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    with open('data/finetune/sysprompt.txt', 'r') as f:
        sys_prompt = f.read()
    sys_prompt = sys_prompt.format(datetime=dt)


    def collate_pad_batch_fn(batch: List[Tuple[List[int], List[int]]]) -> dict:
        MAX_LEN = params.max_position_embeddings
        max_len = max([len(x[0]) for x in batch])
        input_ids = torch.stack([torch.tensor(x[0] + [0] * (max_len - len(x[0]))) for x in batch])
        labels = torch.stack([torch.tensor(x[1] + [CROSS_ENTROPY_IGNORE_IDX] * (max_len - len(x[1]))) for x in batch])
        attention_mask = (input_ids != torch.tensor(0)).long()
        # return input_ids[:, :MAX_LEN], labels[:, :MAX_LEN], attention_mask[:, :MAX_LEN]
        return {"input_ids": input_ids[:, :MAX_LEN],
                "labels": labels[:, :MAX_LEN],
                "attention_mask": attention_mask[:, :MAX_LEN]}


    dataset = DiscordConversations(path="data/finetune/conversations", tokenizer=tokenizer, sys_prompt=sys_prompt)
    dataloader = DataLoader(dataset, batch_size=params.max_batch_size,
                            shuffle=True, collate_fn=collate_pad_batch_fn)

    train(path,
          training_data=dataloader,
          validation_data=dataloader,
          config=params,
          lora_config=lora_params,
          disable_scheduler=False,
          learning_rate=2e-5,
          )
    # validate_model(None, dataloader, True)
