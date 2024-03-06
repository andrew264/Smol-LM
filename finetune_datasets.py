from typing import List, Union, Optional

import datasets
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset

from prompt_format import Prompt


class HFDataset(Dataset):
    def __init__(self, ):
        self.dataset: Optional[datasets.Dataset | pd.DataFrame] = None

    def __len__(self) -> int:
        return len(self.dataset) if self.dataset else 0

    def __getitem__(self, index: int) -> Union[str, List[str]]:
        raise NotImplementedError


class OpenInstruct(HFDataset):
    def __init__(self):
        super().__init__()
        self.dataset = load_dataset("VMware/open-instruct", split='train')

    def __getitem__(self, index: int) -> Union[str, List[str]]:
        row: dict = self.dataset[index]
        prompt = Prompt()
        prompt.add_messages([row['instruction'], row['response']])
        return prompt.get_tokens()


class AlpacaGpt4Dataset(HFDataset):
    def __init__(self, ):
        super().__init__()
        self.dataset = load_dataset("vicgalle/alpaca-gpt4", split='train')

    def __getitem__(self, index: int) -> Union[str, List[str]]:
        row: dict = self.dataset[index]
        instruction = row['instruction'] + '\n' + row['input']
        prompt = Prompt()
        prompt.add_messages([instruction, row['output']])
        return prompt.get_tokens()


class HFnoRobotsDataset(HFDataset):
    def __init__(self, ):
        super().__init__()
        self.dataset = load_dataset("HuggingFaceH4/no_robots", split='train_sft')

    def __getitem__(self, index: int) -> Union[str, List[str]]:
        row: dict = self.dataset[index]
        prompt = Prompt()
        conversation = [c['content'] for c in row['messages']]
        prompt.add_messages(conversation)
        return prompt.get_tokens()


class CSVDataset(HFDataset):
    def __init__(self, path: str, sample_frac=1.0):
        super().__init__()
        self.dataset = pd.read_csv(path).sample(frac=sample_frac, replace=True)

    def __len__(self) -> int:
        return len(self.dataset.index)

    def __getitem__(self, index: int) -> Union[str, List[str]]:
        row = self.dataset.iloc[index]
        prompt = Prompt()
        prompt.add_messages([row.iloc[0], row.iloc[1]])
        return prompt.get_tokens()


class OrcaMath(HFDataset):
    def __init__(self, ):
        super().__init__()
        self.dataset = load_dataset("microsoft/orca-math-word-problems-200k", split='train')

    def __getitem__(self, index: int) -> Union[str, List[str]]:
        row: dict = self.dataset[index]
        prompt = Prompt()
        prompt.add_messages([row['question'], row['answer']])
        return prompt.get_tokens()
