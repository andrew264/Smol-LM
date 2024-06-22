import itertools
from typing import Optional, Callable

from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset

AUDIO_END = "<|audio_end|>"
TEXT_END = "<|end_of_text|>"


class LibreSpeechDataset(IterableDataset):
    def __init__(self, tokenizer: Tokenizer, fe: Optional[Callable] = None,
                 split: str = "train.360", streaming: bool = False):
        super().__init__()
        self._tokenizer = tokenizer
        self._data = load_dataset("/home/andrew264/datasets/librispeech_asr", "clean",
                                  split=split, streaming=streaming, trust_remote_code=True)
        self.feature_extractor = fe

    def __iter__(self):
        for item in self._data:
            sentence = self._tokenizer.encode(AUDIO_END + item['text'].lower() + TEXT_END,
                                              add_special_tokens=False)
            audio = item['audio']['array']
            if self.feature_extractor is not None:
                audio = self.feature_extractor({"audio": audio,
                                                "sampling_rate": item['audio']['sampling_rate']})
            yield {"input_ids": sentence.ids, "attention_mask": sentence.attention_mask, "audio": audio}


class MFCV13(IterableDataset):
    def __init__(self, tokenizer: Tokenizer, fe: Optional[Callable] = None,
                 subset: str = "en", streaming: bool = False):
        super().__init__()
        self._tokenizer = tokenizer
        self._data = load_dataset("/home/andrew264/datasets/common_voice_13_0", name=subset, streaming=streaming,
                                  trust_remote_code=True, num_proc=3, token=True)
        self.feature_extractor = fe

    def __iter__(self):
        for it in itertools.chain(self._data['train'],
                                  self._data['test'],
                                  self._data['validation'],
                                  self._data['other']):
            audio = it['audio']['array']
            if self.feature_extractor is not None:
                audio = self.feature_extractor({"audio": audio,
                                                "sampling_rate": it['audio']['sampling_rate']})
            sentence = self._tokenizer.encode(AUDIO_END + it['sentence'] + TEXT_END,
                                              add_special_tokens=False)
            yield {"input_ids": sentence.ids, "attention_mask": sentence.attention_mask,
                   "audio": audio}
