import os
from random import shuffle
from typing import Optional

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from torch.utils.data import Dataset, IterableDataset


class ParquetDataset(Dataset):
    def __init__(self, directory, tokenizer: Optional[Tokenizer] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
        self.file_row_counts = []
        self.total_rows = 0
        self.cumulative_rows = [0]

        for file in self.parquet_files:
            parquet_file = pq.ParquetFile(file)
            file_row_count = parquet_file.metadata.num_rows
            self.file_row_counts.append(file_row_count)
            self.total_rows += file_row_count
            self.cumulative_rows.append(self.total_rows)

    def __len__(self):
        return self.total_rows

    def __getitem__(self, index):
        if index < 0:
            index = self.total_rows + index

        if not 0 <= index < self.total_rows:
            raise IndexError("Index out of range")

        file_index = self._binary_search(index)
        relative_index = index - self.cumulative_rows[file_index]

        parquet_file = pq.ParquetFile(self.parquet_files[file_index])
        row_group_index = relative_index // parquet_file.metadata.row_group(0).num_rows
        row_group = parquet_file.read_row_group(row_group_index)
        row_in_group = relative_index % row_group.num_rows

        text = row_group.to_pandas().iloc[row_in_group].text
        if self.tokenizer is not None:  # if tokenizer is provided, tokenize the text and return the ids and labels
            tokenized = self.tokenizer.encode(text + "\n</s>", add_special_tokens=False)
            return tokenized.ids, tokenized.ids
        return text

    def _binary_search(self, index):
        left, right = 0, len(self.cumulative_rows) - 1
        while left < right:
            mid = (left + right) // 2
            if self.cumulative_rows[mid] <= index:
                left = mid + 1
            else:
                right = mid
        return left - 1


class ParquetDatasetIter(IterableDataset):
    def __init__(self, directory, tokenizer: Optional[Tokenizer] = None,
                 shuffle_files=True, shuffle_rows=True,
                 max_items=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
        if shuffle_files:
            shuffle(self.parquet_files)
        self.shuffle_rows = shuffle_rows
        self.max_items = max_items

    def __len__(self):
        if self.max_items is not None:
            return self.max_items
        return sum(pq.ParquetFile(file).metadata.num_rows for file in self.parquet_files)

    def __iter__(self):
        items_yielded = 0
        for file in self.parquet_files:
            parquet_file = pq.ParquetFile(file)
            row_groups = list(range(parquet_file.num_row_groups))
            if self.shuffle_rows:
                shuffle(row_groups)
            for row_group_index in row_groups:
                row_group = parquet_file.read_row_group(row_group_index).to_pandas()
                for row in row_group.itertuples():
                    if self.max_items is not None and items_yielded >= self.max_items:
                        return

                    if hasattr(row, 'text'):
                        text = row.text
                        if self.tokenizer is not None:
                            tokenized = self.tokenizer.encode(text + "\n</s>", add_special_tokens=False)
                            yield tokenized.ids, tokenized.ids
                        else:
                            yield text

                        items_yielded += 1
