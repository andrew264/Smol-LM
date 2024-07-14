import glob
from typing import Dict, Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
    def __init__(self, data_dir: str, seq_length: int, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ds = NPDataset(self.data_dir, self.seq_length, True)
        self.val_ds = NPDataset(self.data_dir, self.seq_length, False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def state_dict(self) -> Dict[str, Any]:
        return {"data_dir": self.data_dir, "seq_length": self.seq_length,
                "batch_size": self.batch_size, "num_workers": self.num_workers}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.data_dir = state_dict["data_dir"]
        self.seq_length = state_dict["seq_length"]
        self.batch_size = state_dict["batch_size"]
        self.num_workers = state_dict["num_workers"]
        self.train_ds = NPDataset(self.data_dir, self.seq_length, True)
        self.val_ds = NPDataset(self.data_dir, self.seq_length, False)
