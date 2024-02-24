import os
import time

import datasets
import numpy as np
import tokenizers
import tqdm
from torch.utils.data import DataLoader

PROCESSED_DATA = "./data/processed"

if __name__ == '__main__':
    tokenizer = tokenizers.Tokenizer.from_file("weights/tokenizer.json")
    eot = tokenizer.encode("<|endoftext|>").ids

    cosmopedia = '/mnt/d/Datasets/cosmopedia'
    dataset = datasets.load_dataset(path=cosmopedia, streaming=True, split='train').shuffle(seed=42)
    batch_size = 20_000
    dataloader = DataLoader(dataset, num_workers=6, batch_size=batch_size, shuffle=False)
    length = 31_064_744
    start = time.time()


    def get_batch():
        for batch in dataloader:
            out = []
            encoded_tokens = tokenizer.encode_batch(batch['text'])
            for sample in encoded_tokens:
                out.extend(sample.ids + eot)
            yield out


    split = 'train'
    print(f"Saving {split} to {PROCESSED_DATA}/{split}.bin")
    filename = f"{PROCESSED_DATA}/{split}.bin"
    if os.path.exists(filename):
        os.remove(filename)
    open(filename, "w").close()
    with open(filename, "ab+") as file:
        for tokens in tqdm.tqdm(get_batch(), total=length // batch_size):
            arr = np.array(tokens, dtype=np.uint16)
            arr.tofile(file)
    print(f"Saved {split} to {PROCESSED_DATA}/{split}.bin")

    print(f"Finished in {time.time() - start:.1f}s")
