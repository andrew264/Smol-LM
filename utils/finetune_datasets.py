import glob
import json
from typing import List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from .prompt_format import Role


class DS(Dataset):
    EOT = '</s>'
    CROSS_ENTROPY_IGNORE_IDX = -100

    def __init__(self, tokenizer: Optional[Tokenizer] = None, sys_prompt: Optional[str] = None):
        self._tokenizer = tokenizer
        self._sys_p = sys_prompt
        self._data = None

    def __len__(self) -> int:
        return len(self._data)

    def _get_id_label(self, role: Role, content: str) -> Tuple[List[int], List[int]]:
        match role:
            case Role.SYSTEM:
                prefix = Role.SYSTEM.value
            case Role.USER:
                prefix = "\n" + Role.USER.value
            case Role.ASSISTANT:
                prefix = "\n" + Role.ASSISTANT.value
            case _:
                raise ValueError(f"Invalid role: {role}")

        c = f"{prefix}\n{content.strip()}{self.EOT}"
        enc = self._tokenizer.encode(c, add_special_tokens=False)
        if role in (Role.SYSTEM, Role.USER):
            labels = [self.CROSS_ENTROPY_IGNORE_IDX] * len(enc.ids)
        else:
            labels = enc.ids
        return enc.ids, labels


class HFnoRobotsDataset(DS):
    def __init__(self, tokenizer: Tokenizer, sys_prompt: str, ):
        super().__init__(tokenizer, sys_prompt)
        self._data = load_dataset("HuggingFaceH4/no_robots", split='train_sft')
        self._enc_sys_prompt = self._get_id_label(Role.SYSTEM, sys_prompt)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        row = self._data[idx]
        ids, labels = [], []
        ids.extend(self._enc_sys_prompt[0])
        labels.extend(self._enc_sys_prompt[1])
        for ex in row['messages']:
            role = Role.USER if ex['role'] == 'user' else Role.ASSISTANT
            id_, label = self._get_id_label(role, ex['content'])
            ids.extend(id_)
            labels.extend(label)
        return ids, labels


class CSVDatasetV2(DS):
    def __init__(self, path: str,
                 tokenizer: Tokenizer,
                 sys_prompt: str):
        super().__init__(tokenizer, sys_prompt)
        self._data = pd.read_csv(path)
        self._enc_sys_prompt = self._get_id_label(Role.SYSTEM, sys_prompt)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        row = self._data.iloc[idx]
        ids, labels = [], []
        ids.extend(self._enc_sys_prompt[0])
        labels.extend(self._enc_sys_prompt[1])
        for i in range(0, 2):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            id_, label = self._get_id_label(role, row.iloc[i])
            ids.extend(id_)
            labels.extend(label)
        return ids, labels


class JsonlConversations(DS):
    def __init__(self, path: str,
                 tokenizer: Tokenizer,
                 sys_prompt: str, ):
        super().__init__(tokenizer, sys_prompt)
        self._enc_sys_prompt = self._get_id_label(Role.SYSTEM, sys_prompt)
        self._data = [json.loads(line) for line in open(path, 'r')]

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        row = self._data[idx]
        ids, labels = [], []
        ids.extend(self._enc_sys_prompt[0])
        labels.extend(self._enc_sys_prompt[1])
        for i in range(0, len(row)):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            id_, label = self._get_id_label(role, row[i])
            ids.extend(id_)
            labels.extend(label)
        return ids, labels


class SmallOrca(DS):
    def __init__(self,
                 tokenizer: Tokenizer,
                 sys_prompt: str):
        super().__init__(tokenizer, sys_prompt)
        self._data = load_dataset("prince-canuma/SmallOrca", split='train')

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        data = self._data[idx]
        ids, labels = [], []
        for ex in data['messages']:
            if ex['role'] == 'system':
                role = Role.SYSTEM
            elif ex['role'] == 'user':
                role = Role.USER
            else:
                role = Role.ASSISTANT
            id_, label = self._get_id_label(role, ex['content'])
            ids.extend(id_)
            labels.extend(label)
        return ids, labels


class WizardVicuna(DS):
    def __init__(self, tokenizer: Tokenizer,
                 sys_prompt: str):
        super().__init__(tokenizer, sys_prompt)
        self._data = load_dataset("cognitivecomputations/wizard_vicuna_70k_unfiltered", split='train')
        self._enc_sys_prompt = self._get_id_label(Role.SYSTEM, sys_prompt)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        data = self._data[idx]
        ids, labels = [], []
        ids.extend(self._enc_sys_prompt[0])
        labels.extend(self._enc_sys_prompt[1])
        for ex in data['conversations']:
            if ex['from'] == 'human':
                role = Role.USER
            else:
                role = Role.ASSISTANT
            id_, label = self._get_id_label(role, ex['value'])
            ids.extend(id_)
            labels.extend(label)
        return ids, labels


class DiscordConversations(Dataset):
    EOT = '</s>'
    CROSS_ENTROPY_IGNORE_IDX = -100

    def __init__(self, path: str, tokenizer: Tokenizer, sys_prompt: str):
        self._tokenizer = tokenizer
        self._sys_p = sys_prompt
        self._files = glob.glob(f"{path}/*.json")
        enc_sys_prompt = tokenizer.encode(f"{Role.SYSTEM.value}\n{sys_prompt.strip()}{self.EOT}",
                                          add_special_tokens=False)
        self._enc_sys_prompt = (enc_sys_prompt.ids, [self.CROSS_ENTROPY_IGNORE_IDX] * len(enc_sys_prompt.ids))
        response_head = tokenizer.encode(f"\n{Role.ASSISTANT.value}\n", add_special_tokens=False)
        self._response_head = (response_head.ids, [self.CROSS_ENTROPY_IGNORE_IDX] * len(response_head.ids))

    def __len__(self) -> int:
        return len(self._files)

    def _get_id_label(self, role: str, content: str) -> Tuple[List[int], List[int]]:
        if role == "assistant":
            resp_enc = self._tokenizer.encode(f"{content.strip()}{self.EOT}", add_special_tokens=False)
            ids = self._response_head[0] + resp_enc.ids
            labels = self._response_head[1] + resp_enc.ids
            return ids, labels
        else:
            prompt = f"\n<|{role.strip()}|>\n{content.strip()}{self.EOT}"
            enc = self._tokenizer.encode(prompt, add_special_tokens=False)
            labels = [self.CROSS_ENTROPY_IGNORE_IDX] * len(enc.ids)
            return enc.ids, labels

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        try:
            data = json.load(open(self._files[idx], 'r'))
        except json.decoder.JSONDecodeError as e:
            print(self._files[idx], e)
            raise Exception
        ids, labels = [], []
        ids.extend(self._enc_sys_prompt[0])
        labels.extend(self._enc_sys_prompt[1])
        for ex in data:
            id_, label = self._get_id_label(ex['user'], ex['message'])
            ids.extend(id_)
            labels.extend(label)
        return ids, labels
