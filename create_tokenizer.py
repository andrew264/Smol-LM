import csv
import glob
import os
from collections import Counter
from multiprocessing import pool, Lock

import sentencepiece as spm
import tqdm
from ftfy import fix_text

from utils import yield_lines, clean_text

scrapped_files = glob.glob('data/scrapped/**/*.txt', recursive=True)
BPE_TSV_PATH = "data/bpe_spm.tsv"
BPE_MODEL_PATH = "data/bpe_model"
VOCAB_SIZE = 32000

s = spm.SentencePieceProcessor()

NUM_THREADS = 16
lock = Lock()
counter = Counter()


def process_large_file(file_path: str) -> list[str]:
    for lines in yield_lines(file_path, num_lines=10 ** 6):
        lines = "\n".join(lines)
        lines = fix_text(lines, normalization='NFKC')
        lines = clean_text(lines)
        yield lines.split()


def process_file(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding="latin-1") as f:
        # if f2 size is greater than 100MB, then iterate over the file
        if os.path.getsize(file_path) < 10 ** 8:
            normalized_text = fix_text(f.read(), normalization='NFKC')
            normalized_text = clean_text(normalized_text)
            return normalized_text.split()
        else:
            return []


def create_tsv_for_bpe():
    print("Total scrapped files: ", len(scrapped_files))
    with pool.Pool(NUM_THREADS) as p:
        for _l in tqdm.tqdm(p.imap_unordered(process_file, scrapped_files), total=len(scrapped_files)):
            with lock:
                counter.update(_l)

    large_files = [file for file in scrapped_files if os.path.getsize(file) >= 10 ** 8]
    print("\nTotal large files: ", len(large_files))
    for file in tqdm.tqdm(large_files):
        for _l in process_large_file(file):
            with lock:
                counter.update(_l)

    with open(BPE_TSV_PATH, 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='\t')
        print('Writing to file: ', BPE_TSV_PATH)
        for word in counter:
            writer.writerow([word, counter[word]])


def create_bpe():
    # create_tsv_for_bpe()
    print('Training BPE model: ', BPE_MODEL_PATH)
    spm.SentencePieceTrainer.train(input=BPE_TSV_PATH, model_prefix=BPE_MODEL_PATH, input_format='tsv',
                                   vocab_size=VOCAB_SIZE, hard_vocab_limit=False, model_type='bpe',
                                   pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                                   num_threads=20, character_coverage=0.9999, normalization_rule_name='identity',
                                   split_digits=True, byte_fallback=True)


if __name__ == '__main__':
    create_bpe()
