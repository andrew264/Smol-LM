import os

import torch
from datasets import load_dataset

from transformers import (LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper,
                          RepetitionPenaltyLogitsProcessor, GenerationConfig)
from tokenizers import Tokenizer

from model import ModelConfig, SmolLM, AudioConfig, InternalCache
from utils import get_state_dict_from_safetensors


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file('../weights/tokenizer.json')

config = ModelConfig()
config.hidden_size = 768
config.intermediate_size = 2560
config.num_hidden_layers = 16
config.num_attention_heads = 12
config.max_position_embeddings = 1280
config.num_key_value_heads = 4
config.max_batch_size = 1
config.grad_accumulation_steps = 32

model = SmolLM(config, audio_cfg=AudioConfig()).to(device=device, dtype=torch.bfloat16)
model.eval()

state_dict = get_state_dict_from_safetensors(os.path.join('../weights/test', 'model.safetensors'), device)
model.load_state_dict(state_dict)
del state_dict
model.audio_head._fix_low_precision_training()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)

sample = ds[1]["audio"]
sample_rate, waveform = sample['sampling_rate'], sample['array']
text = ds[1]['text']
print(sample_rate, waveform.shape, text)

processor: LogitsProcessorList = LogitsProcessorList()
processor.append(RepetitionPenaltyLogitsProcessor(1.15))
processor.append(TemperatureLogitsWarper(0.5))
processor.append(TopPLogitsWarper(top_p=0.95))

generation_config: GenerationConfig = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    num_beams=1,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
model.generation_config = generation_config

audio = torch.tensor(waveform, device=device, dtype=torch.float).unsqueeze(0)
input_ids = torch.tensor(tokenizer.encode("nor").ids, device=device).unsqueeze(0)

out = model.generate(input_ids=input_ids,
                     audio=audio,
                     past_key_values=InternalCache(model),
                     logits_processor=processor,
                     generation_config=generation_config, )

print(tokenizer.decode(out[0].cpu().numpy(), skip_special_tokens=True))
