import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, RepetitionPenaltyLogitsProcessor, GenerationConfig, \
    StoppingCriteriaList

from model import ModelConfig, DynamicCache
from prompt_format import Prompt
from utils import load_model, StoppingCriteriaSub

device = torch.device("cuda:0")

if __name__ == '__main__':
    weights = './finetuned-weights/model.safetensors'

    config = ModelConfig.from_json('./weights/config.json')
    config.max_batch_size = 1

    tokenizer = Tokenizer.from_file('./weights/tokenizer.json')
    model = load_model(config, weights, device)
    model.bos_token_id = tokenizer.token_to_id("<s>")

    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    processor.append(RepetitionPenaltyLogitsProcessor(1.05))
    processor.append(TopKLogitsWarper(8))

    generation_config: GenerationConfig = GenerationConfig(
        max_length=512,
        do_sample=True,
        num_beams=1,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cache_implementation=DynamicCache
    )
    model.generation_config = generation_config

    stopping_tokens = [i for i in range(7)]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])


    def multiline_input():
        lines = []
        print('Instruction: ', end="", flush=True)
        while True:
            try:
                line = input()
            except KeyboardInterrupt:
                print()
                break
            if line == '':
                break
            lines.append(line)
        return '\n'.join(lines)


    prompt = Prompt()

    while True:
        inp = multiline_input()
        if inp == '':
            break
        if inp.casefold() == 'reset':
            prompt = Prompt()
            continue

        # prompt
        prompt.add_user_message(inp)
        inp = prompt.get_tokens_for_completion()

        # tokenization
        encoded = tokenizer.encode(inp)
        tokens = torch.tensor(encoded.ids).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0).to(device)

        # generation
        inps = model.prepare_inputs_for_generation(tokens, attention_mask=attention_mask,
                                                   past_key_values=DynamicCache())
        out = model.generate(**inps, logits_processor=processor,
                             generation_config=generation_config,
                             stopping_criteria=stopping_criteria)

        # output
        out_tokens = out[0].tolist()[len(encoded.ids):]
        decoded = tokenizer.decode(out_tokens)
        prompt.add_assistant_message(decoded)
        print(f"Assistant: {decoded}")
