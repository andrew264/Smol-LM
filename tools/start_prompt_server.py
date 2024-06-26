from typing import Optional, Tuple

import torch
from aiohttp import web

from utils import ModelGenerationHandler

device = torch.device("cuda:0")
torch.set_float32_matmul_precision('high')
path = './ft-weights/'

model_handler = ModelGenerationHandler(path, device, 2)


def get_response(input_text, top_p: Optional[float], temp: Optional[float]) -> Tuple[str, int]:
    model_handler.setup_processor(top_p, temp)
    decoded, _, total_toks, _ = model_handler.generate(input_text, max_new_tokens=1024)
    return decoded, total_toks


async def handle(request):
    data = await request.json()
    input_text = data['input']
    top_p = data.get('top_p', 0.90)
    temp = data.get('temp', 1.7)
    output_text, length = get_response(input_text, top_p, temp)

    return web.json_response({'response': output_text, 'cur_length': length,
                              'max_length': model_handler.config.max_position_embeddings, })


def run_server(port=6969):
    model_handler.load_model()
    model_handler.setup_generation()
    model_handler.setup_processor()
    # model_handler.compile_model()
    app = web.Application()
    app.add_routes([web.post('/', handle)])
    web.run_app(app, port=port)


if __name__ == '__main__':
    run_server(port=6969)
