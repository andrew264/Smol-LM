from itertools import cycle

from torch.utils.data import DataLoader


class CyclingDataLoader(object):
    """
    Just a bunch of DataLoaders that cycle through each other.
    """

    def __init__(self, *iterators):
        self.iterators = [iter(it) for it in iterators]
        self.cycle = cycle(self.iterators)

    def __iter__(self):
        return self

    def __len__(self):
        total = 0
        for it in self.iterators:
            if hasattr(it, '__len__'):
                total += len(it)
            else:
                return None
        return total

    def __next__(self):
        if not self.iterators:
            raise StopIteration

        while True:
            current = next(self.cycle)
            try:
                return next(current)
            except StopIteration:
                print(f"Removing exhausted iterator: {current}")
                self.iterators.remove(current)
                if not self.iterators:
                    raise StopIteration
                self.cycle = cycle(self.iterators)
