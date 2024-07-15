from datetime import datetime
from functools import partial
from typing import List, Tuple, Optional

import lightning as L
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

from .conversations import DiscordConversations
from .parquet import ParquetDataset

CROSS_ENTROPY_IGNORE_IDX = -100


def _get_sys_prompt(sys_prompt_path: str) -> str:
    dt = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    with open(sys_prompt_path, 'r') as f:
        sys_prompt = f.read()
    return sys_prompt.format(datetime=dt)


class _CustomFTDataset(Dataset):
    """
    Custom Dataset for fine-tuning.
    Mixes DiscordConversations and ParquetDataset and interleave them.
    the __len__ method returns the len(DiscordConversations) times the mix ratio.
        i.e. if len(DiscordConversations) = 1000 and mix_ratio = 1, then __len__ = 2000 [1 from each dataset]
        if mix_ratio = 2, then __len__ = 3000 [1 from DiscordConversations and 2 from ParquetDataset]
    """

    def __init__(self,
                 tokenizer_path: str,
                 conv_path: str,
                 sys_prompt_path: str,
                 parquet_path: Optional[str] = None,
                 mix_ratio: int = 0):
        super(_CustomFTDataset, self).__init__()
        self.tokenizer = Tokenizer.from_file(path=tokenizer_path)
        self.ds1 = DiscordConversations(path=conv_path,
                                        tokenizer=self.tokenizer,
                                        sys_prompt=_get_sys_prompt(sys_prompt_path))
        if parquet_path is not None:
            self.ds2 = ParquetDataset(directory=parquet_path, tokenizer=self.tokenizer)
        else:
            self.ds2 = []
            mix_ratio = 0
        self.mix_ratio = mix_ratio

    def __len__(self) -> int:
        return len(self.ds1) * (self.mix_ratio + 1)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        total_ratio = self.mix_ratio + 1

        if idx % total_ratio == 0:
            return self.ds1[idx // total_ratio]
        else:
            parquet_idx = (idx // total_ratio) * self.mix_ratio + (idx % total_ratio - 1)
            return self.ds2[parquet_idx % len(self.ds2)]


class CustomFTDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 max_seq_length: int,
                 tokenizer_path: str,
                 conv_path: str,
                 sys_prompt_path: str,
                 parquet_path: Optional[str] = None,
                 mix_ratio: int = 0,
                 max_pad: bool = False):
        """
        Custom DataModule for fine-tuning.
        Mixes DiscordConversations and ParquetDataset and interleave them.
        :param batch_size: Batch size of the Dataloaders
        :param max_seq_length: Maximum allowed Sequence length
        :param tokenizer_path: Path to the tokenizer file.
        :param conv_path: Path to the DiscordConversations dataset.
        :param parquet_path: Path to the ParquetDataset.
        :param sys_prompt_path: Path to the system prompt file.
        :param mix_ratio: Ratio of ParquetDataset to DiscordConversations. Defaults to 1.
        :param max_pad: If True,
        the collate function will pad all sequences to the maximum length supported by the model.
        else, it will pad to the maximum length of the batch.
        """
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.tokenizer_path = tokenizer_path
        self.conv_path = conv_path
        self.parquet_path = parquet_path
        self.sys_prompt_path = sys_prompt_path
        self.mix_ratio = mix_ratio
        self.max_pad = max_pad
        self.train_ds = None
        self.valid_ds = None

    def setup(self, stage: str = None):
        self.train_ds = _CustomFTDataset(tokenizer_path=self.tokenizer_path,
                                         conv_path=self.conv_path,
                                         parquet_path=self.parquet_path,
                                         sys_prompt_path=self.sys_prompt_path,
                                         mix_ratio=self.mix_ratio)
        self.valid_ds = DiscordConversations(path=self.conv_path,
                                             tokenizer=Tokenizer.from_file(path=self.tokenizer_path),
                                             sys_prompt=_get_sys_prompt(self.sys_prompt_path))

    @staticmethod
    def collate_pad_batch_fn(max_len: int, batch: List[Tuple[List[int], List[int]]]) -> dict:
        MAX_LEN = max_len
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

    @staticmethod
    def collate_max_pad_fn(max_len: int, batch: List[Tuple[List[int], List[int]]]) -> dict:
        MAX_LEN = max_len

        input_ids = torch.stack([torch.tensor(x[0] + [0] * (MAX_LEN - len(x[0]))) for x in batch])
        labels = torch.stack([torch.tensor(x[1] + [CROSS_ENTROPY_IGNORE_IDX] * (MAX_LEN - len(x[1]))) for x in batch])
        attention_mask = (input_ids != torch.tensor(0, dtype=input_ids.dtype)).long()

        out = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        return out

    def train_dataloader(self):
        if self.max_pad:
            collate_fn = partial(self.collate_max_pad_fn, self.max_seq_length)
        else:
            collate_fn = partial(self.collate_pad_batch_fn, self.max_seq_length)
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=4)

    def val_dataloader(self):
        if self.max_pad:
            collate_fn = partial(self.collate_max_pad_fn, self.max_seq_length)
        else:
            collate_fn = partial(self.collate_pad_batch_fn, self.max_seq_length)
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=4)
