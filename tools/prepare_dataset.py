import multiprocessing as mp
import os

import numpy as np
import tokenizers
from tqdm import tqdm

from dataset import ParquetDataset

DATA_CACHE_DIR = "../data/processed"
SHARD_SIZE = 2 ** 29

tokenizer = tokenizers.Tokenizer.from_file("../weights/tokenizer.json")
bot = tokenizer.encode("<|begin_of_text|>").ids
eot = tokenizer.encode("<|end_of_text|>").ids
dtype = np.uint32


def tokenize(doc):
    toks = tokenizer.encode(doc['text'], add_special_tokens=False).ids
    return np.array(bot + toks + eot, dtype=dtype)


def write_datafile(f_name: str, toks: np.ndarray):
    """
    Saves token data as a .bin file
    # ik its a bad idea to store binary files without versioning; but idc for now
    """
    print(f"writing {len(toks):,} tokens to {f_name}")
    with open(f_name, "wb") as f:
        f.write(toks.tobytes())


if __name__ == '__main__':
    dataset = ParquetDataset("/home/andrew264/datasets/fineweb-edu/sample/10BT")

    # from llm.c
    nprocs = max(1, os.cpu_count() - 1)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((SHARD_SIZE,), dtype=dtype)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            if token_count + len(tokens) < SHARD_SIZE:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "train"
                filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:03d}.bin")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = SHARD_SIZE - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:03d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])
