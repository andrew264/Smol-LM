from itertools import chain
from typing import List, Union

import pandas as pd
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, Encoding
from torch.utils.data import Dataset

from prompt_format import Prompt


class HFDataset(Dataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        self.tokenized = []
        tokenizer.enable_truncation(max_length=max_length)

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, index: int) -> Union[Encoding, List[Encoding]]:
        return self.tokenized[index]


class OpenInstruct(HFDataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        super().__init__(max_length, tokenizer)
        dataset = load_dataset("VMware/open-instruct", split='train')
        data = []
        for row in dataset:
            prompt = Prompt()
            prompt.add_messages([row['instruction'], row['response']])
            data.append(prompt.get_tokens())
        self.tokenized = tokenizer.encode_batch(data)
        del dataset, data


class AlpacaGpt4Dataset(HFDataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        super().__init__(max_length, tokenizer)
        dataset = load_dataset("vicgalle/alpaca-gpt4", split='train')
        data = []
        for row in dataset:
            instruction = row['instruction'] + '\n' + row['input']
            prompt = Prompt()
            prompt.add_messages([instruction, row['output']])
            data.append(prompt.get_tokens())
        self.tokenized = tokenizer.encode_batch(data)
        del dataset, data


class HFnoRobotsDataset(HFDataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        super().__init__(max_length, tokenizer)
        dataset = load_dataset("HuggingFaceH4/no_robots")
        data = []
        for row in chain.from_iterable([dataset['train_sft'], dataset['test_sft']]):
            prompt = Prompt()
            conversation = [c['content'] for c in row['messages']]
            prompt.add_messages(conversation)
            data.append(prompt.get_tokens())
        self.tokenized = tokenizer.encode_batch(data)
        del dataset, data


class CSVDataset(HFDataset):
    def __init__(self, path: str, max_length: int, tokenizer: Tokenizer, sample_frac=1.0):
        super().__init__(max_length, tokenizer)
        data = []
        df = pd.read_csv(path).sample(frac=sample_frac, replace=True)
        for row in df.itertuples():
            prompt = Prompt()
            prompt.add_messages([row[1], row[2]])
            data.append(prompt.get_tokens())
        self.tokenized = tokenizer.encode_batch(data)
        del df, data


class OrcaMath(HFDataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        super().__init__(max_length, tokenizer)
        dataset = load_dataset("microsoft/orca-math-word-problems-200k", split='train')
        data = []
        for row in dataset:
            prompt = Prompt()
            prompt.add_messages([row['question'], row['answer']])
            data.append(prompt.get_tokens())
        self.tokenized = tokenizer.encode_batch(data)
        del dataset, data


class SortedPadded:
    def __init__(self, dataset: HFDataset, batch_size: int):
        self.dataset = dataset
        self.dataset.tokenized.sort(key=lambda x: len(x.ids))
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch: List[Encoding] = self.dataset[i:i + self.batch_size]
            max_length = max([len(x.ids) for x in batch])
            for i in range(len(batch)):
                batch[i].pad(max_length)
            yield torch.tensor([[x.ids, x.attention_mask] for x in batch])

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        batch = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]
        max_length = max([len(x.ids) for x in batch])
        for i in range(len(batch)):
            batch[i].pad(max_length)
        return torch.tensor([[x.ids, x.attention_mask] for x in batch])

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        batch = self.dataset[self.index:self.index + self.batch_size]
        max_length = max([len(x.ids) for x in batch])
        for i in range(len(batch)):
            batch[i].pad(max_length)
        self.index += self.batch_size
        return torch.tensor([[x.ids, x.attention_mask] for x in batch])
