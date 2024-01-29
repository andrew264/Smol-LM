import pandas as pd
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset

PROMPT_FORMAT = """<|USER|>{instruction}<|endoftext|>
<|ASSISTANT|>{response}<|endoftext|>"""


class HFDataset(Dataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        self.tokenized = []
        tokenizer.enable_padding(pad_id=0, pad_token='<|pad|>', length=max_length + 1)
        tokenizer.enable_truncation(max_length=max_length + 1)

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        ids = self.tokenized[index].ids
        mask = self.tokenized[index].attention_mask
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:]), torch.tensor(mask[:-1])


class DollyDataset(HFDataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        super().__init__(max_length, tokenizer)
        dataset = load_dataset("databricks/databricks-dolly-15k")['train']
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


class AlpacaGpt4Dataset(HFDataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        super().__init__(max_length, tokenizer)
        dataset = load_dataset("vicgalle/alpaca-gpt4")
        data = []
        for row in dataset['train']:
            instruction = row['instruction'] + '\n' + row['input']
            response = row['output']
            data.append(PROMPT_FORMAT.format(instruction=instruction, response=response))
        self.tokenized = tokenizer.encode_batch(data)
        del dataset, data


class HFnoRobotsDataset(HFDataset):
    def __init__(self, max_length: int, tokenizer: Tokenizer):
        super().__init__(max_length, tokenizer)
        dataset = load_dataset("HuggingFaceH4/no_robots")
        data = []
        for row in dataset['train_sft']:
            instruction = row['prompt']
            response = row['messages'][-1]["content"]
            data.append(PROMPT_FORMAT.format(instruction=instruction, response=response))
        self.tokenized = tokenizer.encode_batch(data)
        del dataset, data


class CSVDataset(HFDataset):
    def __init__(self, path: str, max_length: int, tokenizer: Tokenizer, sample_frac=1.0):
        super().__init__(max_length, tokenizer)
        data = []
        df = pd.read_csv(path).sample(frac=sample_frac, replace=True)
        for row in df.itertuples():
            instruction = row[1]
            response = row[2]
            data.append(PROMPT_FORMAT.format(instruction=instruction, response=response))
        self.tokenized = tokenizer.encode_batch(data)
        del df, data
