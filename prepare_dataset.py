import time

import datasets
import numpy as np
import tokenizers
import tqdm

PROCESSED_DATA = "./data/processed"

if __name__ == '__main__':
    num_proc = 20
    tokenizer = tokenizers.Tokenizer.from_file("weights/tokenizer.json")
    eot = tokenizer.encode("<|endoftext|>").ids[0]

    minipile = '/mnt/Ddrive/minipile'
    simple_wikipedia = '/mnt/Ddrive/simple_wikipedia_LM'
    refinedweb = '/mnt/Ddrive/refinedweb-3m'
    d1 = datasets.load_dataset(path=minipile, num_proc=num_proc)['train']
    d2 = datasets.load_dataset(path=simple_wikipedia, num_proc=num_proc)['train']  # id, url, title, text
    # remove the url and id and combine title and text
    d2 = d2.map(lambda x: {'text': x['title'] + '\n' + x['text']}, num_proc=num_proc)
    d2 = d2.remove_columns(['id', 'url', 'title'])

    d3 = datasets.load_dataset(path=refinedweb, num_proc=num_proc)['train']
    dataset = datasets.concatenate_datasets([d1, d2, d3])
    split_dataset = dataset.train_test_split(test_size=0.001, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')


    def enc_batch(x):
        out = {'ids': [], 'len': []}
        encoded = tokenizer.encode_batch(x['text'])
        for item in encoded:
            ids = item.ids + [eot]
            out['ids'].append(ids)
            out['len'].append(len(ids))
        return out


    start = time.time()
    tokenized = split_dataset.map(enc_batch, remove_columns=['text'], num_proc=num_proc,
                                  batched=True, batch_size=500,
                                  desc="Tokenizing")

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f"Saving {split} to {PROCESSED_DATA}/{split}.bin")
        filename = f"{PROCESSED_DATA}/{split}.bin"
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        total_batches = 256

        idx = 0
        for batch_idx in tqdm.tqdm(range(total_batches), desc=f"Writing {split}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])

            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    print(f"Finished in {time.time() - start:.1f}s")
