import torch

from model import Tokenizer, ModelConfig, Transformer

weights = './weights/model_ckpt.pt'
tokenizer_path = './weights/tokenizer.model'
config = './weights/config.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    config = ModelConfig.from_json(config)
    config.max_batch_size = 1

    tokenizer = Tokenizer(tokenizer_path)
    model = Transformer(config)
    checkpoint = torch.load(weights, mmap=False, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.to(dtype=torch.bfloat16, device=device)
    model = model.eval()

    print('model loaded')
    while True:
        prompt = input("Enter a prompt: ")
        if prompt == '':
            break
        tokens = tokenizer.encode(prompt)
        model.generate(torch.tensor(tokens, device=device, dtype=torch.int), tokenizer=tokenizer,
                       max_tokens=350, top_k=8)
