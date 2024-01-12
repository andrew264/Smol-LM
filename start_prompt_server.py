import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, \
    RepetitionPenaltyLogitsProcessor

from model import ModelConfig
from utils import load_model

device = torch.device("cuda")
weights = './finetuned-weights/model_ckpt.pt'

config = ModelConfig.from_json('./weights/config.json')
tokenizer = Tokenizer.from_file('./weights/tokenizer.json')
config.max_batch_size = 1

model = load_model(config, weights, device)

GENERATION_FORMAT = """<|USER|>
{instruction}<|endoftext|>
<|ASSISTANT|>
"""

# Logits processor
processor: LogitsProcessorList = LogitsProcessorList()
processor.append(TemperatureLogitsWarper(0.6))
processor.append(TopKLogitsWarper(40))
processor.append(TopPLogitsWarper(0.90))
processor.append(RepetitionPenaltyLogitsProcessor(1.2))


class MyRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        input_text = urllib.parse.unquote(post_data.decode('utf-8'))

        # Process the input text (You can replace this with your processing logic)
        prompt = GENERATION_FORMAT.format(instruction=input_text)
        prompt = tokenizer.encode(prompt).ids
        prompt = torch.tensor(prompt, dtype=torch.int64, device=device)
        out = model.generate(prompt, max_tokens=128, stream=False, logits_processors=processor)
        output_text = tokenizer.decode(out).strip()

        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(output_text.encode('utf-8'))


def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, MyRequestHandler)
    print(f"Server running on port {port}")
    httpd.serve_forever()


if __name__ == '__main__':
    run_server(port=6969)
