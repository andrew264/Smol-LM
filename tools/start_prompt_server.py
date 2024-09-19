from typing import Tuple

import litserve as ls
import torch

from inference.generation_handler import ModelGenerationHandler


class ModelAPI(ls.LitAPI):
    def __init__(self):
        super().__init__()
        self.device = None
        self.path = None
        self.model_handler = None

    def setup(self, devices: str):
        torch.set_float32_matmul_precision('high')
        self.device = devices
        self.path = './ft-weights/'
        if "cuda" in devices:
            num_beams = 2
            load_in_4bit = False
            merge_lora = True
        else:
            num_beams = 1
            load_in_4bit = True
            merge_lora = False
        self.model_handler = ModelGenerationHandler(self.path, self.device, num_beams)
        self.model_handler.load_model(compiled=False, merge_lora=merge_lora, load_in_4bit=load_in_4bit)


    def decode_request(self, request, **kwargs):
        input_text = request['input']
        top_p = request.get('top_p', 0.99)
        temp = request.get('temp', 1.7)
        return input_text, top_p, temp

    def predict(self, x, **kwargs) -> Tuple[str, int]:
        input_text, top_p, temp = x
        self.model_handler.set_processor(top_p, temp)
        decoded, _, total_toks, _ = self.model_handler.generate(input_text, max_new_tokens=1024)
        return decoded, total_toks

    def encode_response(self, output, **kwargs):
        output_text, length = output
        return {
            'response': output_text,
            'cur_length': length,
            'max_length': self.model_handler.config.max_position_embeddings,
        }


if __name__ == "__main__":
    api = ModelAPI()
    server = ls.LitServer(api)
    server.run(port=6969, generate_client_file=False)
