import time

import datasets
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers, processors

if __name__ == '__main__':
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.ByteLevel(add_prefix_space=False),
                                                       pre_tokenizers.Digits(individual_digits=True)])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=24000,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>",  # Special tokens
                        "<|user|>", "<|assistant|>", "<|system|>",  # Prompt tokens
                        "```", "##", "###", "**"],  # Markdown tokens
        show_progress=True,
        max_token_length=24,
    )

    cosmopedia = '/mnt/d/Datasets/cosmopedia'
    dataset = datasets.load_dataset(path=cosmopedia, split='train', streaming=True).shuffle(seed=42)
    batch_size = 25_000
    length = 31_064_744


    def batch_iterator():
        batch = []
        for sample in dataset:
            batch.append(sample['text'])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


    print("Training tokenizer...")
    start = time.time()
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=length)
    tokenizer.save('./weights/tokenizer.json')
    print(f"Finished training in {(time.time() - start):.2f} seconds")
