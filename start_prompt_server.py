from typing import Optional

import torch
from aiohttp import web
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, RepetitionPenaltyLogitsProcessor, GenerationConfig, \
    StoppingCriteriaList

from model import ModelConfig, DynamicCache
from prompt_format import Prompt
from utils import load_model, StoppingCriteriaSub

device = torch.device("cuda")
weights = './finetuned-weights/model.safetensors'

config = ModelConfig.from_json('./weights/config.json')
tokenizer = Tokenizer.from_file('./weights/tokenizer.json')
config.max_batch_size = 1

model = load_model(config, weights, device)
_eot_token_id = tokenizer.token_to_id("<|endoftext|>")

generation_config: GenerationConfig = GenerationConfig(
    max_length=512,
    do_sample=True,
    num_beams=1,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=_eot_token_id,
    eos_token_id=_eot_token_id,
    cache_implementation=DynamicCache
)
model.generation_config = generation_config

stopping_tokens = [i for i in range(7)]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])


def get_response(input_text, top_k: Optional[int], penalty: Optional[float]):
    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    if penalty is not None and penalty > 0:
        processor.append(RepetitionPenaltyLogitsProcessor(penalty=penalty))
    if top_k is not None and top_k > 0:
        processor.append(TopKLogitsWarper(top_k=top_k))

    prompt = Prompt()
    prompt.add_user_message(input_text)
    encoded = tokenizer.encode(prompt.get_tokens_for_completion())
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
    return decoded


async def handle(request):
    data = await request.json()
    input_text = data['input']
    top_k = data.get('top_k', 10)
    penalty = data.get('penalty', 1.05)
    output_text = get_response(input_text, top_k, penalty)

    return web.json_response({'response': output_text})


def run_server(port=6969):
    app = web.Application()
    app.add_routes([web.post('/', handle)])
    web.run_app(app, port=port)


if __name__ == '__main__':
    run_server(port=6969)
