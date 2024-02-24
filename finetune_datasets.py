import pandas as pd
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset

from prompt_format import Prompt


class HFDataset(Dataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        self.tokenized = []
        tokenizer.enable_truncation(max_length=max_length)

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self.tokenized[index]
        ids = torch.tensor(item.ids)
        mask = torch.tensor(item.attention_mask)
        return ids, mask


class OpenInstruct(HFDataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        super().__init__(max_length, tokenizer)
        dataset = load_dataset("VMware/open-instruct", streaming=True, split='train')
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
        dataset = load_dataset("vicgalle/alpaca-gpt4", streaming=True, split='train')
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
        dataset = load_dataset("HuggingFaceH4/no_robots", streaming=True)
        dataset['train_sft'] = dataset['train_sft'] + dataset['test_sft']
        data = []
        for row in dataset['train_sft']:
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
