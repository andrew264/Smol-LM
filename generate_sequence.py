import os

import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, \
    RepetitionPenaltyLogitsProcessor, GenerationConfig, StoppingCriteriaList

from model import ModelConfig, InternalCache, SmolLM
from utils import get_state_dict_from_safetensors, compile_model, StoppingCriteriaSub

path = './weights'
config = './weights/config.json'
device = torch.device("cuda:0")

if __name__ == '__main__':

    config = ModelConfig.from_json(config)
    config.max_batch_size = 1

    tokenizer = Tokenizer.from_file('weights/tokenizer.json')
    # model
    model_sd = get_state_dict_from_safetensors(os.path.join(path, 'model.safetensors'), device)
    model = SmolLM(config).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(model_sd)
    del model_sd

    # Prepare model
    model.eval()
    compile_model(model)
    torch.cuda.empty_cache()
    model.bos_token_id = tokenizer.token_to_id("<s>")

    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    processor.append(RepetitionPenaltyLogitsProcessor(1.1))
    processor.append(TopKLogitsWarper(12))

    generation_config: GenerationConfig = GenerationConfig(
        max_length=config.max_position_embeddings,
        do_sample=True,
        num_beams=1,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model.generation_config = generation_config

    stopping_tokens = [torch.tensor(128000, device=device), torch.tensor(128001, device=device)]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])

    print('model loaded')
    while True:
        prompt = input("Enter a prompt: ")
        if prompt == '':
            break
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = torch.tensor([encoded_prompt.ids]).to(device)
        attention_mask = torch.tensor([encoded_prompt.attention_mask]).to(device)

        out = model.generate(
            input_ids=tokens,
            attention_mask=attention_mask,
            past_key_values=InternalCache(model),
            logits_processor=processor,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria)
        print(tokenizer.decode(out[0].tolist()))
