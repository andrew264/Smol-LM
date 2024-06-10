import datetime
import os

import torch
from tokenizers import Tokenizer
from transformers import (LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper,
                          RepetitionPenaltyLogitsProcessor, GenerationConfig)

from model import ModelConfig, InternalCache, LoRAConfig, HFNomicEmbeddings, SmolLM  # noqa
from utils import Prompt, inject_lora_adapter, get_state_dict_from_safetensors, compile_model, get_stopping_criteria

device = torch.device("cuda:0")

if __name__ == '__main__':
    path = './ft-weights/'
    num_beams = 1

    config = ModelConfig.from_json(os.path.join(path, 'config.json'))
    config.max_batch_size = num_beams

    tokenizer = Tokenizer.from_file(os.path.join(path, 'tokenizer.json'))

    if os.path.exists(os.path.join(path, 'lora.json')):
        lora_params = LoRAConfig.from_json(os.path.join(path, 'lora.json'))
    else:
        raise ValueError("LoRA config not found.")

    # Load model
    model_sd = get_state_dict_from_safetensors(os.path.join(path, 'model.safetensors'), device)
    model = SmolLM(config).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(model_sd)
    del model_sd

    # Inject LoRA
    adapter_sd = get_state_dict_from_safetensors(os.path.join(path, 'adapter.safetensors'), device)
    model = inject_lora_adapter(model, lora_params, adapter_sd)
    del adapter_sd

    # Prepare model
    model.eval()
    # model = compile_model(model)
    torch.cuda.empty_cache()
    model.bos_token_id = tokenizer.token_to_id("<s>")

    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    processor.append(RepetitionPenaltyLogitsProcessor(1.15))
    processor.append(TemperatureLogitsWarper(0.5))
    processor.append(TopPLogitsWarper(top_p=0.95))

    generation_config: GenerationConfig = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        num_beams=num_beams,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model.generation_config = generation_config

    stopping_criteria = get_stopping_criteria(device)

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
        decoded = tokenizer.decode(out_tokens, skip_special_tokens=False)
        prompt.add_assistant_message(decoded)
        print(f"Assistant: {decoded}")
