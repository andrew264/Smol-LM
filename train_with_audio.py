import itertools
import os
from typing import List, Dict

import torch
import torchaudio.functional as F
from datasets import load_dataset
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, DataLoader

from main import NPDataset
from model import ModelConfig, SmolLMLit
from utils import save_as_safetensors, CyclingDataLoader

model_path = './weights/test/'
tokenizer = Tokenizer.from_file('weights/tokenizer.json')


class LibreSpeechDataset(IterableDataset):
    def __init__(self, split: str = "train.360", streaming: bool = False):
        super().__init__()
        self._data = load_dataset("/home/andrew264/datasets/librispeech_asr", "clean",
                                  split=split, streaming=streaming, trust_remote_code=True)

    def __iter__(self):
        for item in self._data:
            audio = item['audio']['array']
            sentence = tokenizer.encode("<|audio_end|>" + item['text'].lower() + "<|end_of_text|>",
                                        add_special_tokens=False)
            yield {"input_ids": sentence.ids, "attention_mask": sentence.attention_mask, "audio": audio.tolist()}


class MFCV13(IterableDataset):
    def __init__(self, subset: str = "en", streaming: bool = False):
        super().__init__()
        self._data = load_dataset("/home/andrew264/datasets/common_voice_13_0", name=subset, streaming=streaming,
                                  trust_remote_code=True, num_proc=3, token=True)

    def __iter__(self):
        for it in itertools.chain(self._data['train'],
                                  self._data['test'],
                                  self._data['validation'],
                                  self._data['other']):
            audio = it['audio']['array']
            sr = it['audio']['sampling_rate']
            audio = F.resample(torch.tensor(audio), sr, 16000, lowpass_filter_width=8)
            sentence = tokenizer.encode("<|audio_end|>" + it['sentence'] + "<|end_of_text|>",
                                        add_special_tokens=False)
            yield {"input_ids": sentence.ids, "attention_mask": sentence.attention_mask, "audio": audio.tolist()}


config = ModelConfig()
config.vocab_size = tokenizer.get_vocab_size()
config.tie_word_embeddings = True
config.hidden_size = 768
config.intermediate_size = 3072
config.num_hidden_layers = 10
config.num_attention_heads = 12
config.max_position_embeddings = 2048
config.num_key_value_heads = 6
config.max_batch_size = 2
config.grad_accumulation_steps = 50
config.gradient_checkpointing_percent = 0.0
config.has_audio = True
config.lr = 5e-4

MAX_LEN = config.max_position_embeddings


def get_dataloader():
    text_ds = NPDataset("data/processed/fineweb_train_*.bin", MAX_LEN)
    text_dl = DataLoader(text_ds, batch_size=config.max_batch_size, num_workers=4, pin_memory=True)
    audio1_dl = DataLoader(LibreSpeechDataset(), batch_size=config.max_batch_size,
                           collate_fn=collate_pad_audio_batch_fn, num_workers=4, pin_memory=True)
    audio2_dl = DataLoader(MFCV13(), batch_size=config.max_batch_size,
                           collate_fn=collate_pad_audio_batch_fn, num_workers=4, pin_memory=True)

    return CyclingDataLoader(audio1_dl, audio2_dl, text_dl)


def pad_audio_sequence(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)


def collate_pad_audio_batch_fn(batch: List[Dict]) -> Dict:
    audio_list = []
    for b in batch:
        audio_tensor = torch.tensor(b['audio'], dtype=torch.float32)
        audio_tensor = (audio_tensor - audio_tensor.mean()) / audio_tensor.std()
        audio_list.append(audio_tensor)
    audio = pad_audio_sequence(audio_list)

    max_len = max(len(b['input_ids']) for b in batch)
    batch_size = len(batch)
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        length = len(b['input_ids'])
        input_ids[i, :length] = torch.tensor(b['input_ids'], dtype=torch.long)
        attention_mask[i, :length] = torch.tensor(b['attention_mask'], dtype=torch.long)
        labels[i, :length] = torch.tensor(b['input_ids'], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "audio": audio
    }


if __name__ == '__main__':
    model = SmolLMLit(config)
    dataloader = get_dataloader()
    torch.cuda.empty_cache()

    trainer = Trainer(accelerator="gpu",
                      precision="bf16-true",
                      max_epochs=config.epochs,
                      enable_progress_bar=True,
                      log_every_n_steps=10,
                      gradient_clip_val=1.0,
                      accumulate_grad_batches=config.grad_accumulation_steps,
                      default_root_dir=model_path,
                      callbacks=[ModelCheckpoint(dirpath=model_path, save_last=True, filename='last'), ]
                      )

    if os.path.exists(model_path + 'last.ckpt'):
        ckpt_path = model_path + 'last.ckpt'
        print("Resuming training from checkpoint: ", ckpt_path)
    else:
        ckpt_path = None
        print("Starting training from scratch.")

    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=ckpt_path)

    save_as_safetensors(model.state_dict(), os.path.join(model_path, 'model.safetensors'))
