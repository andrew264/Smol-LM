import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopPLogitsWarper, TemperatureLogitsWarper, TopKLogitsWarper, \
    RepetitionPenaltyLogitsProcessor

from model import ModelConfig
from utils import load_model

weights = './weights/model_ckpt.pt'
tokenizer_path = 'weights/tokenizer.json'
config = './weights/config.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    config = ModelConfig.from_json(config)
    config.max_batch_size = 1

    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_model(config, weights, device)

    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    processor.append(TemperatureLogitsWarper(0.6))
    processor.append(TopKLogitsWarper(40))
    processor.append(TopPLogitsWarper(0.90))
    processor.append(RepetitionPenaltyLogitsProcessor(1.2))

    print('model loaded')
    while True:
        prompt = input("Enter a prompt: ")
        if prompt == '':
            break
        tokens = tokenizer.encode(prompt).ids

        model.generate(torch.tensor(tokens, device=device, dtype=torch.int), tokenizer=tokenizer,
                       max_tokens=350, logits_processors=processor)
