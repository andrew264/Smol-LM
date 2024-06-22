import os
from typing import List, Dict

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from dataset import LibreSpeechDataset, MFCV13, NPDataset, CyclingDataLoader
from model import ModelConfig, SmolLMLit, AudioFeatureExtractor
from utils import save_as_safetensors, get_state_dict_from_safetensors

torch.set_float32_matmul_precision('high')
model_path = './weights/test/'
tokenizer = Tokenizer.from_file('weights/tokenizer.json')

config = ModelConfig().from_json('./weights/test/config.json')

MAX_LEN = config.max_position_embeddings


def get_dataloader():
    fe = AudioFeatureExtractor()
    text_ds = NPDataset("data/processed/fineweb_train_*.bin", MAX_LEN)
    text_dl = DataLoader(text_ds, batch_size=config.max_batch_size, num_workers=4, pin_memory=True)
    audio1_dl = DataLoader(LibreSpeechDataset(tokenizer, fe=fe), batch_size=config.max_batch_size,
                           collate_fn=collate_pad_audio_batch_fn, num_workers=4, pin_memory=True)
    audio2_dl = DataLoader(MFCV13(tokenizer, fe=fe), batch_size=config.max_batch_size,
                           collate_fn=collate_pad_audio_batch_fn, num_workers=4, pin_memory=True)

    return CyclingDataLoader(audio1_dl, audio2_dl, text_dl)


def collate_pad_audio_batch_fn(batch: List[Dict]) -> Dict:
    audio = torch.stack([b['audio'] for b in batch])
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
        "audio": audio,
    }


if __name__ == '__main__':
    model = SmolLMLit(config)
    model_sd = get_state_dict_from_safetensors(os.path.join(model_path, 'model.safetensors'), torch.device('cuda'))
    if model_sd is not None:
        model.load_state_dict(model_sd)
    dataloader = get_dataloader()
    torch.cuda.empty_cache()

    checkpoint_callback = ModelCheckpoint(dirpath=model_path, save_last=True, filename='last', every_n_train_steps=5000)
    trainer = Trainer(accelerator="gpu",
                      precision="bf16-true",
                      max_epochs=config.epochs,
                      enable_progress_bar=True,
                      log_every_n_steps=10,
                      gradient_clip_val=1.0,
                      accumulate_grad_batches=config.grad_accumulation_steps,
                      default_root_dir=model_path,
                      callbacks=[checkpoint_callback, ]
                      )

    if os.path.exists(model_path + 'last.ckpt'):
        ckpt_path = model_path + 'last.ckpt'
        print("Resuming training from checkpoint: ", ckpt_path)
    else:
        ckpt_path = None
        print("Starting training from scratch.")

    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=ckpt_path)

    save_as_safetensors(model.state_dict(), os.path.join(model_path, 'model.safetensors'))
