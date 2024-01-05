import torch
from tokenizers import Tokenizer

from model import ModelConfig, Transformer

weights = './weights/model_ckpt.pt'
tokenizer_path = 'weights/tokenizer.json'
config = './weights/config.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    config = ModelConfig.from_json(config)
    config.max_batch_size = 1

    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = Transformer(config)
    model.load_state_dict(torch.load(weights))
    model.to(dtype=torch.bfloat16, device=device)
    model = model.eval()

    print('model loaded')
    while True:
        prompt = input("Enter a prompt: ")
        if prompt == '':
            break
        tokens = tokenizer.encode(prompt).ids
        model.generate(torch.tensor(tokens, device=device, dtype=torch.int), tokenizer=tokenizer,
                       max_tokens=350, top_p=0.9, temperature=0.9)
