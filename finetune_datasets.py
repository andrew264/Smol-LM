from itertools import chain
from typing import List, Union

import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset

from prompt_format import Prompt


class HFDataset(Dataset):
    def __init__(self, ):
        self.data: List[self] = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Union[str, List[str]]:
        return self.data[index]


class OpenInstruct(HFDataset):
    def __init__(self):
        super().__init__()
        dataset = load_dataset("VMware/open-instruct", split='train')
        for row in dataset:
            prompt = Prompt()
            prompt.add_messages([row['instruction'], row['response']])
            self.data.append(prompt.get_tokens())


class AlpacaGpt4Dataset(HFDataset):
    def __init__(self, ):
        super().__init__()
        dataset = load_dataset("vicgalle/alpaca-gpt4", split='train')
        for row in dataset:
            instruction = row['instruction'] + '\n' + row['input']
            prompt = Prompt()
            prompt.add_messages([instruction, row['output']])
            self.data.append(prompt.get_tokens())


class HFnoRobotsDataset(HFDataset):
    def __init__(self, ):
        super().__init__()
        dataset = load_dataset("HuggingFaceH4/no_robots")
        for row in chain.from_iterable([dataset['train_sft'], dataset['test_sft']]):
            prompt = Prompt()
            conversation = [c['content'] for c in row['messages']]
            prompt.add_messages(conversation)
            self.data.append(prompt.get_tokens())


class CSVDataset(HFDataset):
    def __init__(self, path: str, sample_frac=1.0):
        super().__init__()
        df = pd.read_csv(path).sample(frac=sample_frac, replace=True)
        for row in df.itertuples():
            prompt = Prompt()
            prompt.add_messages([row[1], row[2]])
            self.data.append(prompt.get_tokens())


class OrcaMath(HFDataset):
    def __init__(self, ):
        super().__init__()
        dataset = load_dataset("microsoft/orca-math-word-problems-200k", split='train')
        for row in dataset:
            prompt = Prompt()
            prompt.add_messages([row['question'], row['answer']])
            self.data.append(prompt.get_tokens())
