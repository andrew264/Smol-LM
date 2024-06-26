import torch

from utils import ModelGenerationHandler

DEVICE = torch.device("cuda:0")
torch.set_float32_matmul_precision('high')

path = './weights'

if __name__ == '__main__':
    model_handler = ModelGenerationHandler(path, DEVICE, 1)
    model_handler.load_model(compiled=False)

    while True:
        prompt = input("Enter a prompt: ")
        if prompt == '':
            break
        decoded, num_tokens, _, generation_time = model_handler.generate(prompt)
        print(f"Assistant: {decoded}")
        print(f"Generated {num_tokens} tokens in {generation_time:.3f}s")
