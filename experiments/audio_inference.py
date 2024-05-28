import os

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import (LogitsProcessorList, TopKLogitsWarper,
                          GenerationConfig)

from model import ModelConfig, SmolLM, InternalCache
from utils import get_state_dict_from_safetensors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file('../weights/tokenizer.json')

config = ModelConfig()
config.hidden_size = 768
config.intermediate_size = 3072
config.num_hidden_layers = 12
config.num_attention_heads = 24
config.max_position_embeddings = 2048
config.num_key_value_heads = 24
config.max_batch_size = 1

model = SmolLM(config, enable_audio=True).to(device=device, dtype=torch.bfloat16)

state_dict = get_state_dict_from_safetensors(os.path.join('../weights/test', 'model.safetensors'), device)
model.load_state_dict(state_dict)
del state_dict

model.eval()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)

processor: LogitsProcessorList = LogitsProcessorList()
processor.append(TopKLogitsWarper(top_k=10))

generation_config: GenerationConfig = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    num_beams=1,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
model.generation_config = generation_config
for i in range(len(ds)):
    sample = ds[i]["audio"]
    sample_rate, waveform = sample['sampling_rate'], sample['array']
    text = ds[i]['text']
    audio = torch.tensor(waveform, device=device, dtype=torch.bfloat16).unsqueeze(0)
    input_ids = torch.tensor(tokenizer.encode("<s>", add_special_tokens=False).ids, device=device).unsqueeze(0)
    print(text)
    out = model.generate(input_ids=input_ids,
                         audio=audio,
                         past_key_values=InternalCache(model),
                         logits_processor=processor,
                         generation_config=generation_config, )

    print(tokenizer.decode(out[0].cpu().numpy(), skip_special_tokens=True))
    print("_"*50)
