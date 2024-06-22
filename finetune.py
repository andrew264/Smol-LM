import datetime
import os
from typing import Tuple, List

import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from dataset import DiscordConversations, FineWebEdu, CyclingDataLoader
from model import ModelConfig, LoRAConfig
from model import SmolLMLit
from utils import (get_state_dict_from_safetensors, inject_lora_adapter, get_lora_state_dict, save_as_safetensors)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CROSS_ENTROPY_IGNORE_IDX = -100
torch.set_float32_matmul_precision('high')


def train(_path: str,
          training_data: DataLoader,
          validation_data: DataLoader,
          config: ModelConfig,
          lora_config: LoRAConfig,
          use_scheduler: bool = False,
          ):
    # Load model
    model_sd = get_state_dict_from_safetensors(os.path.join(_path, 'model.safetensors'), device)
    model = SmolLMLit(config,
                      use_lora_opt_grp=True,
                      use_scheduler=use_scheduler,
                      ).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(model_sd)
    del model_sd

    # Inject LoRA
    model = inject_lora_adapter(model, lora_config, )

    # Training loop
    torch.cuda.empty_cache()

    trainer = Trainer(accelerator="gpu",
                      precision="bf16-true",
                      max_epochs=config.epochs,
                      enable_progress_bar=True,
                      log_every_n_steps=5,
                      gradient_clip_val=1.0,
                      accumulate_grad_batches=config.grad_accumulation_steps)
    trainer.fit(model, training_data, validation_data)

    # Save model
    adapter_sd = get_lora_state_dict(model)
    save_as_safetensors(adapter_sd, os.path.join(_path, 'adapter.safetensors'))


if __name__ == '__main__':
    from tokenizers import Tokenizer

    tokenizer_path = 'ft-weights/tokenizer.json'
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
        MAX_LEN = 4096
        max_len = max([len(x[0]) for x in batch])

        input_ids = torch.stack([torch.tensor(x[0] + [0] * (max_len - len(x[0]))) for x in batch])
        labels = torch.stack([torch.tensor(x[1] + [CROSS_ENTROPY_IGNORE_IDX] * (max_len - len(x[1]))) for x in batch])
        attention_mask = (input_ids != torch.tensor(0, dtype=input_ids.dtype)).long()

        if len(batch) == 1 and attention_mask.sum().item() == attention_mask.numel():
            attention_mask = None

        out = {
            "input_ids": input_ids[:, :MAX_LEN],
            "labels": labels[:, :MAX_LEN],
        }
        if attention_mask is not None:
            out["attention_mask"] = attention_mask[:, :MAX_LEN]
        return out


    ds1 = DiscordConversations(path="data/finetune/conversations",
                               tokenizer=tokenizer,
                               sys_prompt=sys_prompt, )
    dl1 = DataLoader(ds1, batch_size=params.max_batch_size,
                     shuffle=True, collate_fn=collate_pad_batch_fn, num_workers=4)
    ds2 = FineWebEdu(tokenizer=tokenizer, num_samples=len(ds1), data_shuffle=True)
    dl2 = DataLoader(ds2, batch_size=params.max_batch_size,
                     shuffle=True, collate_fn=collate_pad_batch_fn, num_workers=4)
    dataloader = CyclingDataLoader(dl1, dl2)

    val = DiscordConversations(path="data/finetune/conversations",
                               tokenizer=tokenizer,
                               sys_prompt=sys_prompt, )
    val_data = DataLoader(val, batch_size=params.max_batch_size,
                          shuffle=False, collate_fn=collate_pad_batch_fn, num_workers=4)

    train(path,
          training_data=dataloader,
          validation_data=val_data,
          config=params,
          lora_config=lora_params,
          use_scheduler=True,
          )
