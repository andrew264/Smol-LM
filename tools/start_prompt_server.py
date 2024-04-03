import os
from typing import Optional, Tuple

import torch
from aiohttp import web
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, RepetitionPenaltyLogitsProcessor, GenerationConfig, \
    StoppingCriteriaList

from model import ModelConfig, LoRAConfig, DynamicCache
from utils import StoppingCriteriaSub, load_model

device = torch.device("cuda")
weights = './ft-weights/model.safetensors'

config = ModelConfig.from_json('weights/config.json')
tokenizer = Tokenizer.from_file('weights/tokenizer.json')
config.max_batch_size = 1

if os.path.exists('./ft-weights/lora.json'):
    lora_params = LoRAConfig.from_json('./ft-weights/lora.json')
    print("Loaded LoRA config from file.")
else:
    lora_params = None

model = load_model(config, lora_params, weights, device)
model.bos_token_id = tokenizer.token_to_id("<s>")

generation_config: GenerationConfig = GenerationConfig(
    max_length=config.max_position_embeddings,
    do_sample=True,
    num_beams=12,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    cache_implementation=DynamicCache
)
model.generation_config = generation_config

stopping_tokens = [i for i in range(3)]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])

max_length = config.max_position_embeddings - 128


def get_response(input_text, top_k: Optional[int], penalty: Optional[float]) -> Tuple[str, int]:
    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    if penalty is not None and penalty > 0:
        processor.append(RepetitionPenaltyLogitsProcessor(penalty=penalty))
    if top_k is not None and top_k > 0:
        processor.append(TopKLogitsWarper(top_k=top_k))

    encoded = tokenizer.encode(input_text)
    tokens = torch.tensor(encoded.ids[-max_length:]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded.attention_mask[-max_length:]).unsqueeze(0).to(device)

    # generation
    inps = model.prepare_inputs_for_generation(tokens, attention_mask=attention_mask,
                                               past_key_values=DynamicCache())
    out = model.generate(**inps, logits_processor=processor,
                         generation_config=generation_config,
                         stopping_criteria=stopping_criteria)

    # output
    out_tokens = out[0].tolist()
    decoded = tokenizer.decode(out_tokens[len(tokens[0]):])
    return decoded, len(out_tokens)


async def handle(request):
    data = await request.json()
    input_text = data['input']
    top_k = data.get('top_k', )
    penalty = data.get('penalty', )
    output_text, length = get_response(input_text, top_k, penalty)

    return web.json_response({'response': output_text, 'cur_length': length,
                              'max_length': config.max_position_embeddings, })


def run_server(port=6969):
    app = web.Application()
    app.add_routes([web.post('/', handle)])
    web.run_app(app, port=port)


if __name__ == '__main__':
    run_server(port=6969)
