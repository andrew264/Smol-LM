import os

import lightning as L
import torch
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

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
    has_lora: bool = model.lora_config is not None

    model_sd = get_state_dict_from_safetensors(
        os.path.join(_path, 'model.safetensors'),
        device=torch.device('cpu')
    )
    if model_sd is not None:
        model.load_state_dict(model_sd)
        del model_sd

    if has_lora:
        inject_lora_adapter(model, model.lora_config, merge_lora=False)

    if load_in_8bit:
        model.to_8bit()

    tokenizer_path = 'ft-weights/tokenizer.json'
    sys_prompt_path = 'data/finetune/sysprompt.txt'
    conv_path = 'data/finetune/conversations'
    parquet_path = '/home/andrew264/datasets/fineweb-edu/sample/10BT'
    data_mod = CustomFTDataModule(batch_size=config.max_batch_size,
                                  max_seq_length=config.max_position_embeddings,
                                  tokenizer_path=tokenizer_path,
                                  conv_path=conv_path,
                                  sys_prompt_path=sys_prompt_path,
                                  parquet_path=parquet_path,
                                  mix_ratio=1,
                                  max_pad=False)

    trainer = L.Trainer(accelerator="gpu",
                        precision="bf16-mixed",
                        strategy=DeepSpeedStrategy(
                            stage=3,
                            offload_optimizer=True,
                            cpu_checkpointing=True,
                            partition_activations=True,
                        ),
                        max_epochs=config.epochs,
                        enable_checkpointing=False,
                        enable_progress_bar=True,
                        log_every_n_steps=config.grad_accumulation_steps,
                        gradient_clip_val=1.0,
                        accumulate_grad_batches=config.grad_accumulation_steps)

    trainer.fit(model, datamodule=data_mod)

    if trainer.is_global_zero:
        ckpt_path = os.path.join(_path, 'checkpoint/')
        trainer.save_checkpoint(ckpt_path)
        single_ckpt_path = os.path.join(_path, 'model.pt')
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, single_ckpt_path)
        del trainer, model, data_mod
        torch.cuda.empty_cache()
        if has_lora is not None:
            adapter_sd = get_lora_state_dict(
                torch.load(single_ckpt_path, map_location="cpu")['state_dict']
            )
            save_as_safetensors(adapter_sd, os.path.join(_path, 'adapter.safetensors'))


if __name__ == '__main__':
    model_path = './ft-weights/'

    train(model_path, use_scheduler=True, )
