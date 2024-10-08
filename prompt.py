import datetime

import torch

from inference import ModelGenerationHandler
from utils import Prompt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')


def multiline_input():
    lines = []
    print('Instruction: ', end="", flush=True)
    while True:
        try:
            line = input()
            if line == '':
                break
            lines.append(line)
        except KeyboardInterrupt:
            print()
            break
    return '\n'.join(lines)


def main():
    path = 'ft-weights/'
    if DEVICE.type == 'cuda':
        num_beams = 2
        load_in_4bit = False
    else:
        num_beams = 1
        load_in_4bit = True

    model_handler = ModelGenerationHandler(path, DEVICE, num_beams)
    model_handler.load_model(compiled=False, merge_lora=False, load_in_4bit=load_in_4bit)

    dt = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    with open('data/finetune/sysprompt.txt', 'r') as f:
        sys_prompt = f.read().format(datetime=dt)

    prompt = Prompt(sys_prompt, model_handler.tokenizer)

    while True:
        inp = multiline_input()
        if inp == '':
            break
        if inp.casefold() == 'reset':
            prompt.reset()
            continue

        prompt.add_user_message(inp)
        decoded, num_tokens, _, generation_time = model_handler.generate(prompt.get_tokens_for_completion())
        prompt.add_assistant_message(decoded)

        print(f"Assistant: {decoded}")
        print(f"Generated {num_tokens} tokens in {generation_time:.3f}s ({num_tokens / generation_time:.3f} tokens/s)")


if __name__ == '__main__':
    main()
