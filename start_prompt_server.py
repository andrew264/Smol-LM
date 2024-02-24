from typing import Optional

import torch
from aiohttp import web
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, RepetitionPenaltyLogitsProcessor

from model import ModelConfig
from prompt_format import Prompt
from utils import load_model

device = torch.device("cuda")
weights = './finetuned-weights/accelerator_states/model.safetensors'

config = ModelConfig.from_json('./weights/config.json')
tokenizer = Tokenizer.from_file('./weights/tokenizer.json')
config.max_batch_size = 1

model = load_model(config, weights, device)


def get_response(input_text, top_k: Optional[int], penalty: Optional[float]):
    # Logits processor
    processor: LogitsProcessorList = LogitsProcessorList()
    if penalty is not None and penalty > 0:
        processor.append(RepetitionPenaltyLogitsProcessor(penalty=penalty))
    if top_k is not None and top_k > 0:
        processor.append(TopKLogitsWarper(top_k=top_k))

    # Process the input text (You can replace this with your processing logic)
    prompt = Prompt()
    prompt.add_user_message(input_text)
    prompt = prompt.get_tokens_for_completion(tokenizer)
    prompt = torch.tensor(prompt, dtype=torch.int64, device=device)
    out = model.generate(prompt, max_tokens=1024, stream=False, logits_processors=processor)
    output_text = tokenizer.decode(out).strip()
    return output_text


async def handle(request):
    data = await request.json()
    input_text = data['input']
    top_k = data.get('top_k', None)
    penalty = data.get('penalty', None)
    output_text = get_response(input_text, top_k, penalty)

    return web.json_response({'response': output_text})


def run_server(port=6969):
    app = web.Application()
    app.add_routes([web.post('/', handle)])
    web.run_app(app, port=port)


if __name__ == '__main__':
    run_server(port=6969)
