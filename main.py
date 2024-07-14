import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

from dataset import NPDataModule
from model import SmolLMLit

torch.set_float32_matmul_precision('high')


def train(model_path: str, use_scheduler: bool = False, ):
    checkpoint_path = "checkpoint/last.ckpt"

    model = SmolLMLit(model_path=model_path, use_scheduler=use_scheduler).bfloat16()
    config = model.config

    checkpoint_callback = ModelCheckpoint(dirpath=model_path + "checkpoint/", save_last=True, )

    trainer = L.Trainer(accelerator="gpu",
                        precision="bf16-mixed",
                        strategy=DeepSpeedStrategy(
                            stage=3,
                            offload_optimizer=True,
                            cpu_checkpointing=True,
                            partition_activations=True,
                        ),
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
        single_ckpt_path = "model.pt"
        convert_zero_checkpoint_to_fp32_state_dict(model_path + checkpoint_path, model_path + single_ckpt_path)
        print("Training complete.")


if __name__ == '__main__':
    path = './weights/'

    # training
    train(path)
