import glob
import json
from random import shuffle
from typing import List, Tuple

from tokenizers import Tokenizer
from torch.utils.data import Dataset


class DiscordConversations(Dataset):
    SYSTEM = "<|system|>"
    EOT = '</s>'
    CROSS_ENTROPY_IGNORE_IDX = -100

    def __init__(self, path: str, tokenizer: Tokenizer, sys_prompt: str, shuffle_files: bool = True):
        super().__init__()
        self._tokenizer = tokenizer
        self.assistant_name = "sydney"
        self._files = glob.glob(f"{path}/**/*.json", recursive=True)
        if shuffle_files:
            shuffle(self._files)

        self._enc_sys_prompt = self._encode_with_ignore(f"{self.SYSTEM}{sys_prompt.strip()}\n{self.EOT}")
        self._response_head = self._encode_with_ignore(f"\n<|{self.assistant_name}|>\n")

    def __len__(self) -> int:
        return len(self._files)

    def _encode_with_ignore(self, text: str) -> Tuple[List[int], List[int]]:
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        return encoded.ids, [self.CROSS_ENTROPY_IGNORE_IDX] * len(encoded.ids)

    def _get_id_label(self, role: str, content: str) -> Tuple[List[int], List[int]]:
        if role == self.assistant_name:
            resp_enc = self._tokenizer.encode(f"{content.strip()}\n{self.EOT}", add_special_tokens=False)
            return (
                self._response_head[0] + resp_enc.ids,
                self._response_head[1] + resp_enc.ids
            )
        else:
            return self._encode_with_ignore(f"\n<|{role.strip()}|>\n{content.strip()}\n{self.EOT}")

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        try:
            with open(self._files[idx], 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error in file {self._files[idx]}: {e}")
            raise

        ids, labels = self._enc_sys_prompt
        for ex in data:
            id_, label = self._get_id_label(ex['user'], ex['message'])
            ids.extend(id_)
            labels.extend(label)

        return ids, labels
