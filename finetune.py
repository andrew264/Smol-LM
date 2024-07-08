import os

import lightning as L
import torch

from dataset import CustomFTDataModule
from model import SmolLMLit
from model.peft.utilities import get_lora_state_dict, inject_lora_adapter
from utils import save_as_safetensors, get_state_dict_from_safetensors

torch.set_float32_matmul_precision('high')


def train(_path: str,
          use_scheduler: bool = False,
          load_in_8bit: bool = False):
    model = SmolLMLit(model_path=_path,
                      use_lora_opt_grp=True,
                      use_scheduler=use_scheduler, ).bfloat16()

    config = model.config

    model_sd = get_state_dict_from_safetensors(
        os.path.join(_path, 'model.safetensors'),
        device=torch.device('cpu')
    )
    if model_sd is not None:
        model.load_state_dict(model_sd)
        del model_sd

    if model.lora_config is not None:
        inject_lora_adapter(model, model.lora_config, merge_lora=False)

    if load_in_8bit:
        model.to_8bit()

    data_mod = CustomFTDataModule(batch_size=config.max_batch_size,
                                  max_seq_length=config.max_position_embeddings,
                                  tokenizer_path=tokenizer_path,
                                  conv_path=conv_path,
                                  parquet_path=parquet_path,
                                  sys_prompt_path=sys_prompt_path,
                                  mix_ratio=1)

    trainer = L.Trainer(accelerator="gpu",
                        precision="bf16-mixed",
                        max_epochs=config.epochs,
                        enable_checkpointing=False,
                        enable_progress_bar=True,
                        log_every_n_steps=config.grad_accumulation_steps,
                        gradient_clip_val=1.0,
                        accumulate_grad_batches=config.grad_accumulation_steps)

    trainer.fit(model, datamodule=data_mod)

    # Save adapter weights
    adapter_sd = get_lora_state_dict(model)
    save_as_safetensors(adapter_sd, os.path.join(_path, 'adapter.safetensors'))


if __name__ == '__main__':
    tokenizer_path = 'ft-weights/tokenizer.json'
    sys_prompt_path = 'data/finetune/sysprompt.txt'
    conv_path = 'data/finetune/conversations'
    parquet_path = '/home/andrew264/datasets/fineweb-edu/sample/10BT'
    model_path = './ft-weights/'

    train(model_path, use_scheduler=True, )
