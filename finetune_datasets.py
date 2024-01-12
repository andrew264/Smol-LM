import pandas as pd
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset

PROMPT_FORMAT = """<|USER|>
{instruction}<|endoftext|>
<|ASSISTANT|>
{response}<|endoftext|>"""


class DollyDataset(Dataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        dataset = load_dataset("databricks/databricks-dolly-15k")['train']
        tokenizer.enable_padding(pad_id=0, pad_token='<|pad|>', length=max_length + 1)
        tokenizer.enable_truncation(max_length=max_length + 1)
        data = []
        for row in dataset:
            instruction = row['instruction']
            context = row['context']
            response = row['response']
            if context != '':
                instruction = "Context: " + context + '\n' + instruction
            data.append(PROMPT_FORMAT.format(instruction=instruction, response=response))
        self.tokenized = tokenizer.encode_batch(data)
        del dataset, data

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, index) -> (list[int], list[int]):
        ids = self.tokenized[index].ids
        mask = self.tokenized[index].attention_mask
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:]), torch.tensor(mask[:-1])


class CSVDataset(Dataset):
    def __init__(self, path: str, max_length: int, tokenizer: Tokenizer, sample_frac=1.0):
        tokenizer.enable_padding(pad_id=0, pad_token='<|pad|>', length=max_length + 1)
        tokenizer.enable_truncation(max_length=max_length + 1)
        data = []
        df = pd.read_csv(path).sample(frac=sample_frac, replace=True)
        for row in df.itertuples():
            instruction = row[1]
            response = row[2]
            data.append(PROMPT_FORMAT.format(instruction=instruction, response=response))
        self.tokenized = tokenizer.encode_batch(data)
        del df, data

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, index) -> (list[int], list[int]):
        ids = self.tokenized[index].ids
        mask = self.tokenized[index].attention_mask
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:]), torch.tensor(mask[:-1])


class InstructMixDataset(Dataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        self.dataset = load_dataset("Locutusque/InstructMix-V2")['train']
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding(pad_id=0, pad_token='<|pad|>', length=max_length + 1)
        tokenizer.enable_truncation(max_length=max_length + 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> (list[int], list[int]):
        row = self.dataset[index]
        prompt = PROMPT_FORMAT.format(instruction=row['Input'], response=row['Output'])
        tokenized = self.tokenizer.encode(prompt)
        ids = tokenized.ids
        mask = tokenized.attention_mask
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:]), torch.tensor(mask[:-1])  # x, y, mask
