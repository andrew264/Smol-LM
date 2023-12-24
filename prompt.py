import torch
from tokenizers import Tokenizer

from model import ModelConfig, Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file('./weights/tokenizer.json')


def get_prompt(conv: list[dict[str, str]]) -> list[int]:
    tokens = []
    pad_id, bos_id, eos_id = 0, 1, 2
    for line in conv:
        for k, v in line.items():
            if k == 'SYSTEM':
                tokens += [bos_id] + tokenizer.encode(f" ### System: {v} ",).ids + [eos_id]
            elif k == 'USER':
                tokens += tokenizer.encode(f" ### Instruction: {v} ",).ids + [eos_id]
            elif k == 'ASSISTANT':
                tokens += tokenizer.encode(f" ### Response: {v} ",).ids + [eos_id]
            else:
                raise ValueError(f"Unknown key {k}")
    if conv[-1].keys() == {'USER'}:
        tokens += tokenizer.encode(' ### Response: ',).ids
    return tokens


if __name__ == '__main__':
    weights = './finetuned-weights/model_ckpt.pt'

    config = ModelConfig.from_json('./weights/config.json')
    config.max_batch_size = 1

    model = Transformer(config)
    checkpoint = torch.load(weights, mmap=False, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.to(dtype=torch.bfloat16, device=device)
    model = model.eval()

    CONVERSATION: list[dict[str, str]] = [
        # {"SYSTEM": "You are an AI Assistant. Write response to the given instructions. Follow the instructions carefully."},
    ]


    def multiline_input():
        lines = []
        print('USER: ', end="", flush=True)
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
        CONVERSATION.append({"USER": inp})
        prompt = torch.tensor(get_prompt(CONVERSATION), dtype=torch.int, device=device)
        out = model.generate(prompt, tokenizer=tokenizer, max_tokens=1024, stream=False, temperature=0.9, top_p=0.9)
        CONVERSATION.append({"ASSISTANT": out.strip()})
        print(f"ASSISTANT: {out.strip()}")
