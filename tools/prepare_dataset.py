import os
import time

import datasets
import numpy as np
import tokenizers
import tqdm
from torch.utils.data import DataLoader

PROCESSED_DATA = "../data/processed"

if __name__ == '__main__':
    tokenizer = tokenizers.Tokenizer.from_file("../weights/tokenizer.json")
    bos = tokenizer.encode("<s>").ids
    eot = tokenizer.encode("</s>").ids

    ds = '/home/andrew264/datasets/RedPajama-Data-1T-Sample'
    dataset = datasets.load_dataset(path=ds, streaming=True, split='train', trust_remote_code=True).shuffle(seed=42)
    batch_size = 20_000
    dataloader = DataLoader(dataset, num_workers=6, batch_size=batch_size, shuffle=False)
    start = time.time()


    def get_batch():
        for batch in dataloader:
            out = []
            encoded_tokens = tokenizer.encode_batch(batch['text'], add_special_tokens=False)
            for sample in encoded_tokens:
                out.extend(bos + sample.ids + eot)
            yield out


    split = 'train'
    print(f"Saving {split} to {PROCESSED_DATA}/{split}.bin")
    filename = f"{PROCESSED_DATA}/{split}.bin"
    if os.path.exists(filename):
        os.remove(filename)
    open(filename, "w").close()
    with open(filename, "ab+") as file:
        for tokens in tqdm.tqdm(get_batch()):
            arr = np.array(tokens, dtype=np.uint16)
            arr.tofile(file)
    print(f"Saved {split} to {PROCESSED_DATA}/{split}.bin")

    print(f"Finished in {time.time() - start:.1f}s")
