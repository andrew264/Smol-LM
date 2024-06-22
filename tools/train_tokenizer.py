import os
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers, processors
from torch.utils.data import DataLoader

from dataset import FineWebEdu
from model import ModelConfig

os.environ['TOKENIZERS_PARALLELISM'] = "true"

if __name__ == '__main__':
    path = '../weights/'

    if os.path.exists(path + 'config.json'):
        params = ModelConfig.from_json(path + 'config.json')
        print("Loaded config from file.")
    else:
        params = ModelConfig()
        params.vocab_size = 32000
        print("Created new config.")
        params.to_json(path + 'config.json')

    print(f"Vocab size: {params.vocab_size}")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.ByteLevel(add_prefix_space=False),
                                                       pre_tokenizers.Digits(individual_digits=True)])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=params.vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|begin_of_text|>", "<|end_of_text|>",
                        *["<|reserved_special_token_{}|>".format(i) for i in range(0, 64)],
                        "```", "##", "###", "**"],  # Markdown tokens
        show_progress=True,
        max_token_length=32,
    )

    dataset = FineWebEdu()
    batch_size = 25_000
    dataloader = DataLoader(dataset, batch_size=batch_size,)


    def batch_iterator():
        for batch in dataloader:
            yield batch


    print("Training tokenizer...")
    start = time.time()
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save('./weights/tokenizer.json')
    print(f"Finished training in {(time.time() - start):.2f} seconds")
