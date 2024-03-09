import json
from typing import List, Union, Optional

import pandas as pd
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from prompt_format import Prompt


class DS(Dataset):
    def __init__(self):
        super().__init__()
        self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Union[str, List[str]]:
        return self.data[index]


class HFnoRobotsDataset(DS):
    def __init__(self, sys_prompt: Optional[str] = None,
                 tokenizer: Optional[Union[Tokenizer, PreTrainedTokenizerFast]] = None,
                 max_seq_len=2048):
        super().__init__()
        dataset = load_dataset("HuggingFaceH4/no_robots", split='train_sft')
        prompt = Prompt(sys_prompt, tokenizer)
        for row in dataset:
            conversation = [c['content'] for c in row['messages']]
            prompt.add_messages(conversation)
            if prompt.num_tokens() > max_seq_len:
                if prompt.num_exchanges() == 1:
                    self.data.append(prompt.get_tokens(False))
                    prompt.reset()
                    continue
                prompt.remove_last_exchange()
                self.data.append(prompt.get_tokens(False))
                prompt.reset()
                prompt.add_messages(conversation)
        self.data.append(prompt.get_tokens(False))


class CSVDatasetV2(DS):
    def __init__(self, path: str,
                 sys_prompt: Optional[str] = None,
                 tokenizer: Optional[Union[Tokenizer, PreTrainedTokenizerFast]] = None,
                 max_seq_len=2048):
        super().__init__()
        prompt = Prompt(sys_prompt, tokenizer)
        for _, row in pd.read_csv(path).iterrows():
            prompt.add_messages([row.iloc[0], row.iloc[1]])
            if prompt.num_tokens() > max_seq_len:
                if prompt.num_exchanges() == 1:
                    self.data.append(prompt.get_tokens(False))
                    prompt.reset()
                    continue
                prompt.remove_last_exchange()
                self.data.append(prompt.get_tokens(False))
                prompt.reset()
                prompt.add_messages([row.iloc[0], row.iloc[1]])
        self.data.append(prompt.get_tokens(False))


class JsonlConversations(DS):
    def __init__(self, path: str,
                 sys_prompt: Optional[str] = None,
                 tokenizer: Optional[Union[Tokenizer, PreTrainedTokenizerFast]] = None,
                 max_seq_len=2048):
        super().__init__()
        prompt = Prompt(sys_prompt, tokenizer)
        with open(path, 'r') as f:
            for line in f:
                conversation = json.loads(line)
                conv = [conversation[i:i + 2] for i in range(0, len(conversation), 2)]
                for ex in conv:
                    prompt.add_messages(ex)
                    if prompt.num_tokens() > max_seq_len and prompt.num_exchanges() > 1:
                        self.data.append(prompt.get_tokens(False))
                        prompt.reset()
                        continue
                self.data.append(prompt.get_tokens(False))
                prompt.reset()


class OrcaMath(DS):
    def __init__(self, sys_prompt: Optional[str] = None,
                 tokenizer: Optional[Union[Tokenizer, PreTrainedTokenizerFast]] = None,
                 max_seq_len=2048):
        super().__init__()
        dataset = load_dataset("microsoft/orca-math-word-problems-200k", split='train')
        prompt = Prompt(sys_prompt, tokenizer)
        for row in dataset:
            conversations = [row['question'], row['answer']]
            prompt.add_messages(conversations)
            if prompt.num_tokens() > max_seq_len:
                if prompt.num_exchanges() == 1:
                    self.data.append(prompt.get_tokens(False))
                    prompt.reset()
                    continue
                prompt.remove_last_exchange()
                self.data.append(prompt.get_tokens(False))
                prompt.reset()
                prompt.add_messages(conversations)
        self.data.append(prompt.get_tokens(False))


class WizardVicuna(DS):
    def __init__(self, sys_prompt: Optional[str] = None, ):
        super().__init__()
        dataset = load_dataset("cognitivecomputations/wizard_vicuna_70k_unfiltered", split='train')
        for row in dataset:
            prompt = Prompt(sys_prompt)
            for ex in row['conversations']:
                if ex['from'] == 'human':
                    prompt.add_user_message(ex['value'])
                else:
                    prompt.add_assistant_message(ex['value'])
            self.data.append(prompt.get_tokens(False))
