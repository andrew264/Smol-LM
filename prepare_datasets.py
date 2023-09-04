import datetime
import glob
import json
import multiprocessing as mp
import os
import random

import ftfy
import numpy as np
import tqdm

from model import Tokenizer
from utils import yield_lines, clean_text

BPE_MODEL_PATH = "./weights/tokenizer.model"
PROCESSED = "./data/processed/"
SCRAPPED_DATA = "./data/scrapped/"
TOKENS_PER_FILE = 2 ** 24  # 16M tokens per file
TOTAL_TOKENS = 0
tokenizer = Tokenizer(BPE_MODEL_PATH)


def save_to_file(_tok: list[int]):
    global TOTAL_TOKENS
    TOTAL_TOKENS += len(_tok)
    _tok = np.array(_tok, dtype=np.uint16)
    filename = PROCESSED + str(int(datetime.datetime.now().timestamp())) + ".bin"
    with open(filename, 'wb') as out_file:
        out_file.write(_tok.tobytes())
    print(f"\nSaved {filename}")


def _process_file(path: str) -> list[int]:
    _c = ""
    for lines in yield_lines(path):
        lines = "\n".join(lines)
        lines = ftfy.fix_text(lines, normalization='NFKC')
        _c += clean_text(lines) + "\n"
    return tokenizer.encode(_c, bos=True)


if __name__ == '__main__':
    if not os.path.exists(PROCESSED):
        os.makedirs(PROCESSED)

    # glob all scrapped text data
    scrapped_files = glob.glob(SCRAPPED_DATA + '**/*.txt', recursive=True)
    random.shuffle(scrapped_files)
    print("Total scrapped files:- ", len(scrapped_files))
    tokens = []
    with mp.Pool(14) as pool:
        for _f in tqdm.tqdm(pool.imap_unordered(_process_file, scrapped_files), total=len(scrapped_files)):
            tokens.extend(_f)
            while len(tokens) > TOKENS_PER_FILE:
                save_to_file(tokens[:TOKENS_PER_FILE])
                tokens = tokens[TOKENS_PER_FILE:]

    # glob all scrapped json data
    scrapped_files = glob.glob(SCRAPPED_DATA + "/**/*.json", recursive=True)
    random.shuffle(scrapped_files)
    print("\nTotal scrapped files:- ", len(scrapped_files))
    for file in scrapped_files:
        list_of_text = json.load(open(file, 'r', encoding='utf-8'))
        for content in tqdm.tqdm(range(0, len(list_of_text), 25000)):
            encoded: list[list[int]] = tokenizer.encode(list_of_text[content:content + 25000], bos=True)
            for item in encoded:
                tokens.extend(item)
            while len(tokens) > TOKENS_PER_FILE:
                save_to_file(tokens[:TOKENS_PER_FILE])
                tokens = tokens[TOKENS_PER_FILE:]
    if len(tokens) > 0:
        save_to_file(tokens)

    print(f"Total tokens: {TOTAL_TOKENS} ({(TOTAL_TOKENS / 1e09):.4f} billion)")
