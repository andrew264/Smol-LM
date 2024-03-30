from tokenizers import Tokenizer

from model import ModelConfig
from utils import load_model, save_model

config_path = "../weights/config.json"
tokenizer_path = "../weights/tokenizer.json"
weights_path = "../weights/model.safetensors"

if __name__ == '__main__':
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.add_special_tokens(["<|assistant|>", "<|system|>", "<|", "|>"])
    params = ModelConfig.from_json(config_path)
    model = load_model(params, None, weights_path)
    model.resize_embeddings(tokenizer.get_vocab_size())
    save_model(model, weights_path)
    params.vocab_size = model.vocab_size
    params.to_json(config_path)
    tokenizer.save(tokenizer_path)
    print("Done")
