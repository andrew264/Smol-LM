import datasets
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers, processors

if __name__ == '__main__':
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.ByteLevel(add_prefix_space=True),
                                                       pre_tokenizers.Digits(individual_digits=False)])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|pad|>", "<|endoftext|>", "<|USER|>", "<|SYSTEM|>", "<|ASSISTANT|>"],
        show_progress=True,
        max_token_length=64,
    )
    dataset_path = '/run/media/andrew264/nvme1n1p5/minipile'
    dataset = datasets.load_dataset(path=dataset_path, num_proc=20)['train']


    def batch_iterator(batch_size=100000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]


    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset), )
    tokenizer.save('./weights/tokenizer.json')
    print("Saved tokenizer.json")
