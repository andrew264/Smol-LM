import time

import datasets
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers, processors, normalizers

if __name__ == '__main__':
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.ByteLevel(add_prefix_space=True),
                                                       pre_tokenizers.Digits(individual_digits=True)])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), ])
    trainer = trainers.BpeTrainer(
        vocab_size=16000,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|pad|>", "<|endoftext|>", "<|USER|>", "<|SYSTEM|>", "<|ASSISTANT|>"],
        show_progress=True,
        max_token_length=16,
    )

    num_proc = 20
    minipile = '/mnt/d/minipile'
    refinedweb = '/mnt/d/refinedweb-3m'
    d1 = datasets.load_dataset(path=minipile, num_proc=num_proc)['train']
    d2 = datasets.load_dataset("BEE-spoke-data/wikipedia-20230901.en-deduped", "text-only", num_proc=num_proc)
    d3 = datasets.load_dataset(path=refinedweb, num_proc=num_proc)['train']
    dataset = datasets.concatenate_datasets([d3, d2['train'], d2['validation'], d2['test'], d1])


    def batch_iterator(batch_size=10000):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]["text"]
            yield batch


    print("Training tokenizer...")
    start = time.time()
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset), )
    tokenizer.save('./weights/tokenizer.json')
    print(f"Finished training in {(time.time() - start):.2f} seconds")
