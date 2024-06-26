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

    def __init__(self, path: str,
                 tokenizer: Tokenizer,
                 sys_prompt: str,
                 shuffle_files: bool = True,
                 ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._sys_p = sys_prompt
        self.assistant_name = "sydney"
        self._files = glob.glob(f"{path}/**/*.json")
        if shuffle_files:
            shuffle(self._files)
        enc_sys_prompt = tokenizer.encode(f"{self.SYSTEM}{sys_prompt.strip()}\n{self.EOT}",
                                          add_special_tokens=False)
        self._enc_sys_prompt = (enc_sys_prompt.ids, [self.CROSS_ENTROPY_IGNORE_IDX] * len(enc_sys_prompt.ids))
        response_head = tokenizer.encode(f"\n<|{self.assistant_name}|>\n", add_special_tokens=False)
        self._response_head = (response_head.ids, [self.CROSS_ENTROPY_IGNORE_IDX] * len(response_head.ids))

    def __len__(self) -> int:
        return len(self._files)

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
