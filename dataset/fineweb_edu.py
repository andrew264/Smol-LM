from random import randint
from typing import Optional

from datatrove.pipeline.readers import ParquetReader
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset


class FineWebEdu(IterableDataset):
    EOT = '</s>'

    def __init__(self,
                 tokenizer: Optional[Tokenizer] = None,
                 path: str = "/home/andrew264/datasets/fineweb-edu",
                 num_samples: Optional[int] = None,
                 data_shuffle: bool = True,
                 ):
        self._tokenizer = tokenizer
        self._data = ParquetReader(path,
                                   glob_pattern="sample/10BT/*.parquet",
                                   read_metadata=False,
                                   shuffle_files=True)()

        self.num_samples = num_samples
        self._num_samp_iter = 0
        self.data_shuffle = data_shuffle

    def __len__(self) -> Optional[int]:
        return self.num_samples

    def __iter__(self):
        if self._tokenizer is None:
            # let's just iterate over the data and return the text
            for item in self._data:
                yield item.text
        else:
            if self.data_shuffle:  # poor man's random access
                for _ in range(randint(5, 100)):
                    next(self._data)
            self._num_samp_iter += 1
            if self.num_samples is not None and self._num_samp_iter > self.num_samples:
                raise StopIteration
            tokenized = self._tokenizer.encode(next(self._data).text + f"\n{self.EOT}", add_special_tokens=False)
            yield tokenized.ids, tokenized.ids
