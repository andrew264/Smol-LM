from typing import Tuple

import litserve as ls
import torch

from utils import ModelGenerationHandler


class ModelAPI(ls.LitAPI):
    def __init__(self):
        super().__init__()
        self.device = None
        self.path = None
        self.model_handler = None

    def setup(self, device):
        torch.set_float32_matmul_precision('high')
        self.device = torch.device("cuda:0")
        self.path = './ft-weights/'
        self.model_handler = ModelGenerationHandler(self.path, self.device, 2)
        self.model_handler.load_model(compiled=False)

    def decode_request(self, request, **kwargs):
        input_text = request['input']
        top_p = request.get('top_p', 0.99)
        temp = request.get('temp', 1.7)
        return input_text, top_p, temp

    def predict(self, inputs, **kwargs) -> Tuple[str, int]:
        input_text, top_p, temp = inputs
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
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=6969)
