import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, \
    RepetitionPenaltyLogitsProcessor, GenerationConfig

from model import ModelConfig, DynamicCache
from utils import load_model

weights = './weights/model.safetensors'
tokenizer_path = 'weights/tokenizer.json'
config = './weights/config.json'
device = torch.device("cuda:0")

if __name__ == '__main__':

    config = ModelConfig.from_json(config)
    config.max_batch_size = 1

    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_model(config, weights, device)
    _eot_token_id = tokenizer.token_to_id("</s>")
    model.bos_token_id = tokenizer.token_to_id("<s>")

    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    processor.append(RepetitionPenaltyLogitsProcessor(1.1))
    processor.append(TopKLogitsWarper(8))

    generation_config: GenerationConfig = GenerationConfig(
        max_length=350,
        do_sample=True,
        num_beams=1,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cache_implementation=DynamicCache
    )
    model.generation_config = generation_config

    print('model loaded')
    while True:
        prompt = input("Enter a prompt: ")
        if prompt == '':
            break
        encoded_prompt = tokenizer.encode(f"{prompt}")
        tokens = torch.tensor(encoded_prompt.ids).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoded_prompt.attention_mask).unsqueeze(0).to(device)
        kv_cache = DynamicCache()

        inps = model.prepare_inputs_for_generation(tokens, attention_mask=attention_mask,
                                                   past_key_values=kv_cache)
        out = model.generate(**inps, logits_processor=processor, generation_config=generation_config, )
        print(tokenizer.decode(out[0].tolist()))
