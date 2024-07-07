import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset import NPDataModule
from model import SmolLMLit
from utils import save_as_safetensors

torch.set_float32_matmul_precision('high')


def train(model_path: str, use_scheduler: bool = False, ):
    model = SmolLMLit(model_path,
                      use_scheduler=use_scheduler, )

    config = model.config

    datamodule = NPDataModule("data/processed/fineweb_train_*.bin",
                              seq_length=config.max_position_embeddings,
                              batch_size=config.max_batch_size)

    checkpoint_callback = ModelCheckpoint(dirpath=model_path, save_last=True, filename='last')

    trainer = L.Trainer(accelerator="gpu",
                        precision="bf16-mixed",
                        max_epochs=config.epochs,
                        enable_progress_bar=True,
                        val_check_interval=5000,
                        limit_val_batches=500,
                        log_every_n_steps=10,
                        gradient_clip_val=1.0,
                        accumulate_grad_batches=config.grad_accumulation_steps,
                        default_root_dir=model_path,
                        callbacks=[checkpoint_callback],
                        )

    if os.path.exists(model_path + 'last.ckpt'):
        ckpt_path = model_path + 'last.ckpt'
        print("Resuming training from checkpoint: ", ckpt_path)
    else:
        ckpt_path = None
        print("Starting training from scratch.")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if config.tie_word_embeddings:
        # safetensors do not support shared memory, so we need to save the weights separately
        model.model.embed_tokens.weight = torch.nn.Parameter(model.model.embed_tokens.weight.clone())

    save_as_safetensors(model.state_dict(), os.path.join(model_path, 'model.safetensors'))

    print("Training complete.")


if __name__ == '__main__':
    path = './weights/'

    # training
    train(path)
