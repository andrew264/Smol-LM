import datetime
import os

import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, RepetitionPenaltyLogitsProcessor, GenerationConfig, \
    StoppingCriteriaList

from model import ModelConfig, InternalCache, LoRAConfig, HFNomicEmbeddings  # noqa
from utils import Prompt, StoppingCriteriaSub, load_model

device = torch.device("cuda:0")

if __name__ == '__main__':
    weights = './ft-weights/model.safetensors'
    num_beams = 2

    config = ModelConfig.from_json('./ft-weights/config.json')
    config.max_batch_size = num_beams

    tokenizer = Tokenizer.from_file('./weights/tokenizer.json')

    if os.path.exists('./ft-weights/lora.json'):
        lora_params = LoRAConfig.from_json('./ft-weights/lora.json')
        print("Loaded LoRA config from file.")
    else:
        lora_params = None

    model = load_model(config, lora_config=lora_params, path=weights, device=device)
    torch.compile(model=model.forward, fullgraph=True, mode='max-autotune')
    model.bos_token_id = tokenizer.token_to_id("<s>")

    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    processor.append(RepetitionPenaltyLogitsProcessor(1.05))
    processor.append(TopKLogitsWarper(12))

    generation_config: GenerationConfig = GenerationConfig(
        max_length=config.max_position_embeddings,
        do_sample=True,
        num_beams=num_beams,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model.generation_config = generation_config

    stopping_tokens = [i for i in range(3)]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])

    dt = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    with open('data/finetune/sysprompt.txt', 'r') as f:
        sys_prompt = f.read()
    sys_prompt = sys_prompt.format(datetime=dt)


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


    # embedder = HFNomicEmbeddings()
    # vec_db = './data/rag_chroma'
    embedder, vec_db = None, None
    prompt = Prompt(sys_prompt, tokenizer, embeddings_model=embedder, vector_store_path=vec_db)

    while True:
        inp = multiline_input()
        if inp == '':
            break
        if inp.casefold() == 'reset':
            prompt.reset()
            continue

        # prompt
        prompt.add_user_message(inp)
        inp = prompt.get_tokens_for_completion()

        # tokenization
        encoded = tokenizer.encode(inp, add_special_tokens=False)
        tokens = torch.tensor([encoded.ids]).to(device)
        attention_mask = torch.tensor([encoded.attention_mask]).to(device)

        # generation
        out = model.generate(input_ids=tokens,
                             attention_mask=attention_mask,
                             logits_processor=processor,
                             past_key_values=InternalCache(model),
                             generation_config=generation_config,
                             stopping_criteria=stopping_criteria)

        # output
        out_tokens = out[0].tolist()[len(encoded.ids):]
        decoded = tokenizer.decode(out_tokens)
        prompt.add_assistant_message(decoded)
        print(f"Assistant: {decoded}")
