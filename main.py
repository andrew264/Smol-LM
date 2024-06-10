import glob
import os

import lightning as L
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from model import ModelConfig
from model.lightning_model import SmolLMLit
from utils import (compile_model, save_as_safetensors)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')


class NPDataset(Dataset):
    def __init__(self, _path, seq_length=1024, train: bool = True, dtype=np.uint32):
        file_paths = glob.glob(_path)
        if train:
            file_paths = file_paths[:-1]
        else:
            file_paths = [file_paths[-1]]

        self.seq_length = seq_length
        self.dtype = dtype
        self.memmaps = []
        self.sizes = []
        self.cumulative_sizes = []

        total_size = 0

        for file_path in file_paths:
            memmap = np.memmap(file_path, dtype=dtype, mode='r')
            num_samples = len(memmap) // seq_length
            memmap = memmap[:num_samples * seq_length].reshape(num_samples, seq_length)

            self.sizes.append(num_samples)
            self.cumulative_sizes.append(total_size)
            total_size += num_samples
            self.memmaps.append(memmap)

        self.total_size = total_size
        self.cumulative_sizes.append(total_size)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index) -> dict:
        if index < 0:
            index += self.total_size
        if index >= self.total_size or index < 0:
            raise IndexError("Index out of range")

        file_index = np.searchsorted(self.cumulative_sizes, index, side='right') - 1
        internal_index = index - self.cumulative_sizes[file_index]

        item = torch.from_numpy(np.int32(self.memmaps[file_index][internal_index]))
        return {"input_ids": item, "labels": item}


class NPDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, seq_length: int, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.train_ds = NPDataset(self.data_dir, self.seq_length, True)
        self.val_ds = NPDataset(self.data_dir, self.seq_length, False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)


def train(model_path: str, datamod: L.LightningDataModule, config: ModelConfig,
          use_scheduler: bool = False, ):
    model = SmolLMLit(config,
                      use_scheduler=use_scheduler,
                      ).to(device=device, dtype=torch.bfloat16)
    model = compile_model(model)

    checkpoint_callback = ModelCheckpoint(dirpath=model_path, save_last=True, filename='last')
    trainer = Trainer(accelerator="gpu",
                      precision="bf16-true",
                      max_epochs=config.epochs,
                      enable_progress_bar=True,
                      val_check_interval=5000,
                      limit_val_batches=500,
                      log_every_n_steps=10,
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
    trainer.fit(model, datamodule=datamod, ckpt_path=ckpt_path)

    if config.tie_word_embeddings:
        # safetensors do not support shared memory, so we need to save the weights separately
        model.model.embed_tokens.weight = torch.nn.Parameter(model.model.embed_tokens.weight.clone())

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
    datamod = NPDataModule("data/processed/fineweb_train_*.bin",
                           seq_length=params.max_position_embeddings,
                           batch_size=params.max_batch_size)
    train(path,
          datamod=datamod,
          config=params,
          )
