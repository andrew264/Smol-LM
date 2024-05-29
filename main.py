import os
from typing import Optional

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from model import ModelConfig
from model.lightning_model import SmolLMLit
from utils import (compile_model, save_as_safetensors)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')


def move_to_device(batch):
    return {k: v.to(device) for k, v in batch.items()}


class NPDataset(Dataset):
    def __init__(self, _path, block_size=1024, validation_split=False):
        validation_size = 1000 * block_size
        self.data = np.memmap(_path, dtype=np.uint16, mode='r')
        if validation_split:
            self.data = self.data[-validation_size:]
        else:
            self.data = self.data[:-validation_size]
        self.num_samples = len(self.data) // block_size
        self.data = self.data[:self.num_samples * block_size].reshape(self.num_samples, block_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> dict:
        item = np.int64(self.data[index])
        return {"input_ids": item, "labels": item}


def train(model_path: str, training_data: DataLoader, config: ModelConfig, validation_data: Optional[DataLoader] = None,
          use_scheduler: bool = False, ):
    model = SmolLMLit(config,
                      use_scheduler=use_scheduler,
                      ).to(device=device, dtype=torch.bfloat16)
    compile_model(model)

    checkpoint_callback = ModelCheckpoint(dirpath=model_path, save_last=True, filename='last')
    trainer = Trainer(accelerator="gpu",
                      precision="bf16-true",
                      max_epochs=config.epochs,
                      enable_progress_bar=True,
                      val_check_interval=2000,
                      log_every_n_steps=2,
                      gradient_clip_val=1.0,
                      accumulate_grad_batches=config.grad_accumulation_steps,
                      default_root_dir=model_path,
                      callbacks=[checkpoint_callback],
                      )
    if os.path.exists(model_path + 'last.ckpt'):
        ckpt_path = model_path + 'last.ckpt'
        print("Resuming training from checkpoint: ", ckpt_path)
    else:
        ckpt_path = None
        print("Starting training from scratch.")
    trainer.fit(model, training_data, validation_data, ckpt_path=ckpt_path)

    save_as_safetensors(model.state_dict(), os.path.join(model_path, 'model.safetensors'))

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

    train_data = DataLoader(training, batch_size=params.max_batch_size, shuffle=False, drop_last=True, num_workers=4)
    val_data = DataLoader(validation, batch_size=params.max_batch_size, shuffle=False, drop_last=True,
                          pin_memory=True, num_workers=4)
    print("Train: ", len(train_data), "Validation: ", len(val_data))

    train(path,
          training_data=train_data,
          validation_data=val_data,
          config=params,
          )
