import torch
from tokenizers import Tokenizer

from model import ModelConfig, Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file('./weights/tokenizer.json')

GENERATION_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response: <|startoftext|>
"""

if __name__ == '__main__':
    weights = './finetuned-weights/model_ckpt.pt'

    config = ModelConfig.from_json('./weights/config.json')
    config.max_batch_size = 1

    model = Transformer(config)
    checkpoint = torch.load(weights, mmap=False, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.to(dtype=torch.bfloat16, device=device)
    model = model.eval()


    def multiline_input():
        lines = []
        print('Instruction: ', end="", flush=True)
        while True:
            line = input()
            if line == '':
                break
            lines.append(line)
        return '\n'.join(lines)


    while True:
        inp = multiline_input()
        if inp == '':
            break
        prompt = GENERATION_FORMAT.format(instruction=inp)
        prompt = tokenizer.encode(prompt).ids
        prompt = torch.tensor(prompt, dtype=torch.int64, device=device)
        out = model.generate(prompt, max_tokens=1024, stream=False, temperature=1.0, top_p=0.9)
        out = tokenizer.decode(out)
        print(f"Response: {out.strip()}")
