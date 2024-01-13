import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, RepetitionPenaltyLogitsProcessor

from model import ModelConfig
from utils import load_model

device = torch.device("cuda")
tokenizer = Tokenizer.from_file('./weights/tokenizer.json')

GENERATION_FORMAT = """<|USER|>{instruction}<|endoftext|>
<|ASSISTANT|>"""

if __name__ == '__main__':
    weights = './finetuned-weights/model_ckpt.pt'

    config = ModelConfig.from_json('./weights/config.json')
    config.max_batch_size = 1

    model = load_model(config, weights, device)

    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    processor.append(RepetitionPenaltyLogitsProcessor(1.1))
    processor.append(TopKLogitsWarper(20))


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
        out = model.generate(prompt, max_tokens=512, stream=False, logits_processors=processor)
        out = tokenizer.decode(out)
        print(f"Response: {out.strip()}")
