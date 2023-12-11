import torch

from model import ModelConfig, Tokenizer, Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer('./weights/tokenizer.model')


def get_prompt(conv: list[dict[str, str]]) -> list[int]:
    tokens = []
    for line in conv:
        for k, v in line.items():
            if k == 'SYSTEM':
                tokens += tokenizer.encode(f"###{k}: {v}\n", bos=True, eos=True)
            elif k == 'USER':
                tokens += tokenizer.encode(f"\n###{k}: {v}\n", eos=True, bos=True if not tokens else False)
            elif k == 'ASSISTANT':
                tokens += tokenizer.encode(f"\n###{k}: {v}\n", eos=True, bos=False)
            else:
                raise ValueError(f"Unknown key {k}")
    if conv[-1].keys() == {'USER'}:
        tokens += tokenizer.encode('\n###ASSISTANT:', eos=False, bos=False)
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
        {"SYSTEM": "You are an AI Assistant. Respond to USER prompts."},
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
        out = model.generate(prompt, tokenizer=tokenizer, max_tokens=128, stream=False, temperature=1.0, top_k=20)
        CONVERSATION.append({"ASSISTANT": out.strip()})
        print(f"ASSISTANT: {out.strip()}")
