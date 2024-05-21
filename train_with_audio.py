import os
import time
from typing import List, Tuple

import bitsandbytes as bnb
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, DataLoader

from model import ModelConfig, SmolLM, AudioConfig
from utils import count_parameters, compile_model, save_as_safetensors
from main import NPDataset

model_path = './weights/test'


class LibreSpeechDataset(IterableDataset):
    def __init__(self, tokenizer: Tokenizer, split: str = "train.100", streaming: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self._data = load_dataset("/home/andrew264/datasets/librispeech_asr", "clean",
                                  split=split, streaming=streaming, trust_remote_code=True)

    def __iter__(self) -> iter:
        for item in self._data:
            audio = item['audio']['array']
            sentence = self.tokenizer.encode(item['text'] + "</s>")
            yield audio.tolist(), sentence.ids


config = ModelConfig()
config.hidden_size = 768
config.intermediate_size = 2560
config.num_hidden_layers = 12
config.num_attention_heads = 12
config.max_position_embeddings = 1024
config.num_key_value_heads = 4
config.max_batch_size = 4
config.grad_accumulation_steps = 32

MAX_LEN = config.max_position_embeddings


def collate_pad_audio_batch_fn(batch: List[Tuple[List[np.ndarray], List[int]]]) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    audio, sentence = zip(*batch)
    # pad audio
    max_audio_len = max(len(a) for a in audio)
    audio = [torch.tensor(a + [0] * (max_audio_len - len(a))) for a in audio]
    audio = torch.stack(audio)
    # pad sentence
    max_len = max([len(x) for x in sentence])
    input_ids = torch.stack([torch.tensor(x + [0] * (max_len - len(x))) for x in sentence])
    labels = torch.stack([torch.tensor(x + [-100] * (max_len - len(x))) for x in sentence])
    return input_ids[:, :MAX_LEN], labels[:, :MAX_LEN], audio


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SmolLM(config, audio_cfg=AudioConfig()).to(device=device, dtype=torch.bfloat16)
model.audio_head._fix_low_precision_training()

tokenizer = Tokenizer.from_file('weights/tokenizer.json')

count_parameters(model)
# total_steps = len(dataset)

accelerator = Accelerator(gradient_accumulation_steps=config.grad_accumulation_steps)
compile_model(model)
model = accelerator.prepare_model(model)
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)
optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)

accumulated_loss = 0
start_step = 0
start_time, time_delta = time.time(), 0.
print_step = max(100, config.grad_accumulation_steps)
model.train()
print(
    f"Tokens per batch: {config.max_batch_size * config.grad_accumulation_steps * config.max_position_embeddings}")

# print(f"Training Step: {start_step} of {total_steps} | {start_step / total_steps * 100:.2f}%")
torch.cuda.empty_cache()
i = 0
text_ds = NPDataset('./data/processed/train.bin', config.max_position_embeddings)
text_dl = DataLoader(text_ds, batch_size=config.max_batch_size, )
text_iter = iter(text_dl)
audio_dl = DataLoader(LibreSpeechDataset(tokenizer=tokenizer), batch_size=config.max_batch_size,
                      collate_fn=collate_pad_audio_batch_fn)

for item in audio_dl:
    i += 2
    sentence, labels, audio = item
    sentence = sentence.to(device)
    labels = labels.to(device)
    audio = audio.to(device)
    with accelerator.accumulate(model):
        loss = model(input_ids=sentence, labels=labels, attention_mask=None, audio=audio).loss
    accumulated_loss += loss.item()
    accelerator.backward(loss)  # backward pass
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad()

    # text batch
    try:
        item = next(text_iter)
    except StopIteration:
        item = None
    if item is not None:
        ids = item[0].to(device)
        with accelerator.accumulate(model):
            loss = model(input_ids=ids, labels=ids).loss
        accumulated_loss += loss.item()
        accelerator.backward(loss)
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

    if i % 5000 == 0:
        save_as_safetensors(model.state_dict(), os.path.join(model_path, 'model.safetensors'))
