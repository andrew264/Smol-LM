import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, \
    RepetitionPenaltyLogitsProcessor

from model import ModelConfig, Transformer
from utils import load_model

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

    model = load_model(config, weights, device)

    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    processor.append(TemperatureLogitsWarper(0.6))
    processor.append(TopKLogitsWarper(40))
    processor.append(TopPLogitsWarper(0.90))
    processor.append(RepetitionPenaltyLogitsProcessor(1.2))


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
        out = model.generate(prompt, max_tokens=1024, stream=False, logits_processors=processor)
        out = tokenizer.decode(out)
        print(f"Response: {out.strip()}")
