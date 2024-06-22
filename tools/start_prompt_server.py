import os
from typing import Optional, Tuple

import torch
from aiohttp import web
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopPLogitsWarper

from model import ModelConfig, LoRAConfig, InternalCache, SmolLM, TemperatureRangeLogitsWarper
from utils import inject_lora_adapter, get_state_dict_from_safetensors, get_stopping_criteria, \
    get_generation_config

device = torch.device("cuda:0")
dtype = torch.bfloat16
path = './ft-weights/'

config = ModelConfig.from_json(os.path.join(path, 'config.json'))
tokenizer = Tokenizer.from_file(os.path.join(path, 'tokenizer.json'))
config.max_batch_size = 1

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

generation_config = get_generation_config(2000)
stopping_criteria = get_stopping_criteria(device=device)
model.generation_config = generation_config

max_length = config.max_position_embeddings - 128


def get_response(input_text, top_p: Optional[float], temp: Optional[float]) -> Tuple[str, int]:
    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    if top_p is not None and top_p > 0:
        processor.append(TopPLogitsWarper(top_p=top_p))
    if temp is not None and temp > 0:
        processor.append(TemperatureRangeLogitsWarper(temp, 0.8, 12))

    encoded = tokenizer.encode(input_text)
    tokens = torch.tensor(encoded.ids[-max_length:]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded.attention_mask[-max_length:]).unsqueeze(0).to(device)

    # generation
    out = model.generate(
        input_ids=tokens,
        attention_mask=attention_mask,
        past_key_values=InternalCache(model, dtype=dtype),
        logits_processor=processor,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria
    )

    # output
    out_tokens = out[0].tolist()
    for bad in [[523, 28766], [28789, 28766]]:
        if bad == out_tokens[-len(bad):]:
            out_tokens = out_tokens[:-len(bad)]
    decoded = tokenizer.decode(out_tokens[len(tokens[0]):])
    return decoded, len(out_tokens)


async def handle(request):
    data = await request.json()
    input_text = data['input']
    top_p = data.get('top_p', 0.90)
    temp = data.get('temp', 1.7)
    output_text, length = get_response(input_text, top_p, temp)

    return web.json_response({'response': output_text, 'cur_length': length,
                              'max_length': config.max_position_embeddings, })


def run_server(port=6969):
    app = web.Application()
    app.add_routes([web.post('/', handle)])
    web.run_app(app, port=port)


if __name__ == '__main__':
    run_server(port=6969)
