import datasets
import numpy as np
import tokenizers
import tqdm

PROCESSED_DATA = "./data/processed"

if __name__ == '__main__':
    num_proc = 18
    tokenizer = tokenizers.Tokenizer.from_file("weights/tokenizer.json")
    eot = tokenizer.encode("<|endoftext|>").ids[0]

    dataset_path = '/run/media/andrew264/nvme1n1p5/openwebtext'
    dataset = datasets.load_dataset(path=dataset_path, num_proc=num_proc)
    split_dataset = dataset['train'].train_test_split(test_size=0.0005, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')


    def enc(x):
        ids = tokenizer.encode(x['text']).ids
        ids.append(eot)
        out = {'ids': ids, 'len': len(ids)}
        return out


    tokenized = split_dataset.map(enc, remove_columns=['text'], num_proc=num_proc, desc="Tokenizing")

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f"Saving {split} to {PROCESSED_DATA}/{split}.bin")
        filename = f"{PROCESSED_DATA}/{split}.bin"
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm.tqdm(range(total_batches), desc=f"Writing {split}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])

            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
