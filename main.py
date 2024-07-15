import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset import NPDataModule
from model import SmolLMLit
from utils import save_as_safetensors

torch.set_float32_matmul_precision('high')


def train(model_path: str, use_scheduler: bool = False, ):
    checkpoint_path = "checkpoint/last.ckpt"

    model = SmolLMLit(model_path=model_path, use_scheduler=use_scheduler).bfloat16()
    config = model.config

    checkpoint_callback = ModelCheckpoint(dirpath=model_path + "checkpoint/", save_last=True, )

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

    datamodule = NPDataModule("data/processed/fineweb_train_*.bin",
                              seq_length=config.max_position_embeddings,
                              batch_size=config.max_batch_size)

    ckpt_exists = os.path.exists(model_path + checkpoint_path)
    trainer.fit(model, datamodule=datamodule, ckpt_path=model_path + checkpoint_path if ckpt_exists else None)

    if trainer.is_global_zero:
        trainer.save_checkpoint(model_path + checkpoint_path)
        if config.tie_word_embeddings:
            # safetensors do not support shared weights, so we need to clone the weights
            model.model.embed_tokens.weight = torch.nn.Parameter(model.model.embed_tokens.weight.clone())
        save_as_safetensors(model.state_dict(),
                            os.path.join(model_path, 'model.safetensors'))
        print("Training complete.")


if __name__ == '__main__':
    path = './weights/'

    # training
    train(path)
