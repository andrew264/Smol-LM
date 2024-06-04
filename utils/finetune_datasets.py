import glob
import json
from typing import List, Optional, Tuple

from datatrove.pipeline.readers import ParquetReader
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
        self.assistant_name = "sydney"


class DiscordConversations(Dataset):
    EOT = '</s>'
    CROSS_ENTROPY_IGNORE_IDX = -100

    def __init__(self, path: str,
                 tokenizer: Tokenizer,
                 sys_prompt: str,
                 pt_data_mix: int = 2,  # as this number increases, the ft data will be spread apart
                 validation: bool = False):
        self._tokenizer = tokenizer
        self._sys_p = sys_prompt
        self.assistant_name = "sydney"
        self._files = glob.glob(f"{path}/**/*.json")
        if not validation:
            if pt_data_mix < 2:
                raise ValueError("pt_data_mix should be at least 2 else there will be consequences")
            self._pt_data = ParquetReader("/home/andrew264/datasets/fineweb-edu",
                                          glob_pattern="sample/10BT/*.parquet",
                                          read_metadata=False,
                                          shuffle_files=True)()
            self.pt_data_mix = pt_data_mix
        self.validation = validation
        enc_sys_prompt = tokenizer.encode(f"{Role.SYSTEM.value}{sys_prompt.strip()}\n{self.EOT}",
                                          add_special_tokens=False)
        self._enc_sys_prompt = (enc_sys_prompt.ids, [self.CROSS_ENTROPY_IGNORE_IDX] * len(enc_sys_prompt.ids))
        response_head = tokenizer.encode(f"\n<|{self.assistant_name}|>\n", add_special_tokens=False)
        self._response_head = (response_head.ids, [self.CROSS_ENTROPY_IGNORE_IDX] * len(response_head.ids))

    def __len__(self) -> int:
        return len(self._files) * self.pt_data_mix if not self.validation else len(self._files)

    def _get_id_label(self, role: str, content: str) -> Tuple[List[int], List[int]]:
        if role == self.assistant_name:
            resp_enc = self._tokenizer.encode(f"{content.strip()}\n{self.EOT}", add_special_tokens=False)
            ids = self._response_head[0] + resp_enc.ids
            labels = self._response_head[1] + resp_enc.ids
            return ids, labels
        else:
            prompt = f"\n<|{role.strip()}|>\n{content.strip()}\n{self.EOT}"
            enc = self._tokenizer.encode(prompt, add_special_tokens=False)
            labels = [self.CROSS_ENTROPY_IGNORE_IDX] * len(enc.ids)
            return enc.ids, labels

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        if not self.validation and idx % self.pt_data_mix != 0:
            tokenized = self._tokenizer.encode(next(self._pt_data).text + f"\n{self.EOT}", add_special_tokens=False)
            return tokenized.ids, tokenized.ids
        if not self.validation:
            idx = idx // self.pt_data_mix
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
