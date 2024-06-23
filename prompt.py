import datetime

import torch

from utils import Prompt, ModelGenerationHandler

DEVICE = torch.device("cuda:0")
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
    path = './ft-weights/'
    num_beams = 2

    model_handler = ModelGenerationHandler(path, DEVICE, num_beams)
    model_handler.load_model()
    model_handler.setup_generation()
    model_handler.setup_processor()
    model_handler.compile_model()

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
        print(f"Generated {num_tokens} tokens in {generation_time:.3f}s")


if __name__ == '__main__':
    main()
