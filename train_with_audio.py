import itertools
import os
import time
from typing import List

import bitsandbytes as bnb
import torch
from accelerate import Accelerator
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, DataLoader
import torchaudio.functional as F

from main import NPDataset
from model import ModelConfig, SmolLM, AudioConfig
from utils import count_parameters, compile_model, save_as_safetensors, CyclingDataLoader, \
    get_state_dict_from_safetensors

model_path = './weights/test'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LibreSpeechDataset(IterableDataset):
    def __init__(self, tokenizer: Tokenizer, split: str = "train.360", streaming: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self._data = load_dataset("/home/andrew264/datasets/librispeech_asr", "clean",
                                  split=split, streaming=streaming, trust_remote_code=True)

    def __iter__(self):
        for item in self._data:
            audio = item['audio']['array']
            sentence = self.tokenizer.encode("<s>" + item['text'].lower() + "</s>")
            yield {"input_ids": sentence.ids, "audio": audio.tolist()}


class MFCV13(IterableDataset):
    def __init__(self, tokenizer: Tokenizer, subset: str = "en", streaming: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self._data = load_dataset("/home/andrew264/datasets/common_voice_13_0", name=subset, streaming=streaming,
                                  trust_remote_code=True, num_proc=3, token=True)

    def __iter__(self):
        for item in itertools.chain(self._data['train'],
                                    self._data['test'],
                                    self._data['validation'],
                                    self._data['other']):
            audio = item['audio']['array']
            sr = item['audio']['sampling_rate']
            audio = F.resample(torch.tensor(audio, device=device), sr, 16000, lowpass_filter_width=6)
            sentence = self.tokenizer.encode("<s>" + item['sentence'] + "</s>")
            yield {"input_ids": sentence.ids, "audio": audio.tolist()}


def move_to_device(batch):
    return {k: v.to(device) for k, v in batch.items()}


config = ModelConfig()
config.hidden_size = 768
config.intermediate_size = 2560
config.num_hidden_layers = 16
config.num_attention_heads = 12
config.max_position_embeddings = 1280
config.num_key_value_heads = 4
config.max_batch_size = 4
config.grad_accumulation_steps = 32

MAX_LEN = config.max_position_embeddings


def collate_pad_audio_batch_fn(batch: List[dict]) -> dict:
    # pad audio
    max_audio_len = max(len(a['audio']) for a in batch)
    audio = torch.stack([torch.tensor(a['audio'] + [0] * (max_audio_len - len(a['audio']))) for a in batch])
    # pad sentence
    max_len = max([len(x['input_ids']) for x in batch])
    input_ids = torch.stack([torch.tensor(x['input_ids'] + [0] * (max_len - len(x['input_ids']))) for x in batch])
    labels = torch.stack([torch.tensor(x['input_ids'] + [-100] * (max_len - len(x['input_ids']))) for x in batch])
    return {"input_ids": input_ids, "labels": labels, "audio": audio}


model = SmolLM(config, audio_cfg=AudioConfig()).to(device=device, dtype=torch.bfloat16)
model.audio_head._fix_low_precision_training()

tokenizer = Tokenizer.from_file('weights/tokenizer.json')

count_parameters(model)

accelerator = Accelerator(gradient_accumulation_steps=config.grad_accumulation_steps)
compile_model(model)
model = accelerator.prepare_model(model)
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)

if os.path.exists(os.path.join(model_path, 'model.safetensors')):
    state_dict = get_state_dict_from_safetensors(os.path.join(model_path, 'model.safetensors'), device)
    model.load_state_dict(state_dict)
    del state_dict
    print("Loaded model weights.")

torch.cuda.empty_cache()
text_ds = NPDataset('./data/processed/train.bin', config.max_position_embeddings)
text_dl = DataLoader(text_ds, batch_size=config.max_batch_size, )
audio1_dl = DataLoader(LibreSpeechDataset(tokenizer=tokenizer), batch_size=config.max_batch_size,
                       collate_fn=collate_pad_audio_batch_fn)
audio2_dl = DataLoader(MFCV13(tokenizer=tokenizer), batch_size=config.max_batch_size,
                       collate_fn=collate_pad_audio_batch_fn)

dataloaders = CyclingDataLoader(audio1_dl, audio2_dl, text_dl)

accumulated_loss = 0
start_time, time_delta = time.time(), 0.
print_step = max(250, config.grad_accumulation_steps)
model.train()
print(f"Tokens per batch: {config.max_batch_size * config.grad_accumulation_steps * config.max_position_embeddings}")

i = 0
for item in dataloaders:
    with accelerator.accumulate(model):
        loss = model(**move_to_device(item)).loss
    accumulated_loss += loss.item()
    accelerator.backward(loss)  # backward pass
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad()

    if i % print_step == 0 and i > 0:
        time_delta = time.time() - start_time
        avg_loss = accumulated_loss / print_step
        avg_perplexity = torch.exp(torch.tensor(avg_loss))
        batch_per_sec = print_step / time_delta

        accelerator.print(f"Step: {i} | Loss: {avg_loss:.3f} | Perplexity: {avg_perplexity:.3f} | "
                          f"Elapsed Time: {time_delta:.1f}s | Batch/sec: {batch_per_sec:.1f}")

        start_time = time.time()
        accumulated_loss = 0

    if i % 5000 == 0 and i > 0:
        save_as_safetensors(model.state_dict(), os.path.join(model_path, 'model.safetensors'))

    i += 1

save_as_safetensors(model.state_dict(), os.path.join(model_path, 'model.safetensors'))
