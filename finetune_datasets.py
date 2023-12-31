import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset
import pandas as pd

PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that completes the request.

<|USER|>
{instruction}

<|ASSISTANT|>
{response}
<|endoftext|>"""


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
        tokenized = tokenizer.encode_batch(data)
        self.data = torch.tensor(data=[x.ids for x in tokenized], dtype=torch.int64)
        del dataset, data, tokenized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> (list[int], list[int]):
        return self.data[index][:-1], self.data[index][1:]


class CSVDataset(Dataset):
    def __init__(self, path: str, max_length: int, tokenizer: Tokenizer):
        tokenizer.enable_padding(pad_id=0, pad_token='<|pad|>', length=max_length + 1)
        tokenizer.enable_truncation(max_length=max_length + 1)
        data = []
        df = pd.read_csv(path)
        for row in df.itertuples():
            instruction = row[1]
            response = row[2]
            data.append(PROMPT_FORMAT.format(instruction=instruction, response=response))
        tokenized = tokenizer.encode_batch(data)
        self.data = torch.tensor(data=[x.ids for x in tokenized], dtype=torch.int64)
        del df, data, tokenized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> (list[int], list[int]):
        return self.data[index][:-1], self.data[index][1:]
