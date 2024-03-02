import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, \
    RepetitionPenaltyLogitsProcessor

from model import ModelConfig
from utils import load_model

weights = './weights/model.safetensors'
tokenizer_path = 'weights/tokenizer.json'
config = './weights/config.json'
device = torch.device("cuda:0")

if __name__ == '__main__':

    config = ModelConfig.from_json(config)
    config.max_batch_size = 1

    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_model(config, weights, device)
    _eot_token_id = tokenizer.token_to_id("<|endoftext|>")

    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    processor.append(RepetitionPenaltyLogitsProcessor(1.1))
    processor.append(TopKLogitsWarper(8))

    print('model loaded')
    while True:
        prompt = input("Enter a prompt: ")
        if prompt == '':
            break
        tokens = [_eot_token_id] + tokenizer.encode(prompt).ids

        model.generate(torch.tensor(tokens, device=device, dtype=torch.int64), tokenizer=tokenizer,
                       max_tokens=350, logits_processors=processor, )
