from abc import ABC
from itertools import cycle

from torch.utils.data import IterableDataset, Dataset


class InterleaveDataset(IterableDataset, ABC):
    """
    Just a bunch of Datasets that are interleaved.
    """

    def __init__(self, *datasets: IterableDataset | Dataset):
        self.datasets = datasets
        self.cycle = cycle([iter(it) for it in datasets])

    def __len__(self):
        total = 0
        for it in self.datasets:
            if hasattr(it, '__len__'):
                total += it.__len__()
            else:
                return None
        return total

    def __next__(self):
        while True:
            current = next(self.cycle)
            try:
                return next(current)
            except StopIteration:
                pass

    def __iter__(self):
        return self
