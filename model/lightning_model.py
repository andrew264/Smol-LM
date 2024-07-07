import os
from functools import partial
from typing import Optional, List

import bitsandbytes as bnb
import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import get_cosine_schedule_with_warmup

from utils.utils import get_state_dict_from_safetensors
from .block import Block
from .config import ModelConfig, LoRAConfig
from .norm import get_rmsnorm_class
from .peft.utilities import inject_lora_adapter
from .rotary import RotaryEmbedding

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss

    CrossEntropyLoss = partial(CrossEntropyLoss, inplace_backward=True)
except ImportError:
    from torch.nn import CrossEntropyLoss


def get_optimizer_grouped_parameters(model: nn.Module, weight_decay: float) -> list[dict]:  # from llm.c repo
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    return optim_groups


def get_lora_plus_optimizer_group(model: nn.Module,
                                  lr: float,
                                  lr_ratio: int = 4,
                                  lr_embedding: float = 1e-6,
                                  ) -> List[dict]:
    param_groups = {
        "groupA": {},
        "groupB": {},
        "embedding": {},
    }
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'embed_tokens' in param_name:
            param_groups["embedding"][param_name] = param
        elif 'lora_A' in param_name:
            param_groups["groupA"][param_name] = param
        elif 'lora_B' in param_name:
            param_groups["groupB"][param_name] = param

    optimizer_grouped_parameters = [
        {"params": param_groups["groupA"].values(), "lr": lr},
        {"params": param_groups["groupB"].values(), "lr": lr * lr_ratio},  # learn the B group faster than A
        {"params": param_groups["embedding"].values(), "lr": lr_embedding},
    ]
    return optimizer_grouped_parameters


class SmolLMLit(L.LightningModule):
    def __init__(self,
                 model_path: str,
                 use_lora_opt_grp: bool = False,
                 use_scheduler: bool = False):
        super().__init__()
        if os.path.exists(model_path + 'config.json'):
            config = ModelConfig.from_json(model_path + 'config.json')
        else:
            raise FileNotFoundError(f"Config file not found at {model_path + 'config.json'}")
        self.lora_config = None
        if os.path.exists(model_path + 'lora.json'):
            self.lora_config = LoRAConfig.from_json(model_path + 'lora.json')
        self.model_path = model_path
        self.config = config
        self.tie_word_embeddings = config.tie_word_embeddings
        self.checkpointing_layers = config.checkpointing_layers
        self.max_length = config.max_position_embeddings
        self.use_lora_opt_grp = use_lora_opt_grp
        self.use_scheduler = use_scheduler
        self.model = None
        self.rotary_emb = None
        self.lm_head = None
        self.loss_fn = None

    def configure_model(self):
        config = self.config
        self.model = nn.ModuleDict(dict(
            embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            layers=nn.ModuleList(Block(config, idx) for idx in range(config.num_hidden_layers)),
            norm=get_rmsnorm_class()(config.hidden_size, eps=config.rms_norm_eps),
        ))

        self.rotary_emb = RotaryEmbedding(dim=config.hidden_size // config.num_attention_heads,
                                          base=config.rope_theta, )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        self.loss_fn = CrossEntropyLoss(ignore_index=-100)

        self.apply(self._init_weights)  # to initialize weights

        model_sd = get_state_dict_from_safetensors(os.path.join(self.model_path, 'model.safetensors'))
        if model_sd is not None:
            self.load_state_dict(model_sd)
            del model_sd

        if self.lora_config is not None:
            inject_lora_adapter(self, self.lora_config, merge_lora=False)
        self.to(self.device)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def configure_optimizers(self):
        lr = self.config.lr
        if self.use_lora_opt_grp:
            optimizer = bnb.optim.AdamW8bit(
                params=get_lora_plus_optimizer_group(self, lr=lr),
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )
        else:
            optimizer = bnb.optim.AdamW8bit(
                params=get_optimizer_grouped_parameters(self, weight_decay=0.1),
                lr=lr,
                betas=(0.9, 0.95),
            )
        if self.use_scheduler:
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=5,
                                                        num_training_steps=self.trainer.estimated_stepping_batches)

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
        return optimizer

    @staticmethod
    def _generate_causal_mask(attention_mask, input_tensor):
        if attention_mask is None:
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        causal_mask = torch.full((sequence_length, sequence_length),
                                 fill_value=min_dtype, dtype=dtype, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

        return causal_mask

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
    ):

        x = self.model.embed_tokens(input_ids)

        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        freqs = self.rotary_emb(position_ids)

        causal_mask = self._generate_causal_mask(attention_mask, x)

        for i, layer in enumerate(self.model.layers):
            if self.training and i in self.checkpointing_layers:
                x = checkpoint(layer,
                               x, freqs, causal_mask,
                               use_reentrant=False, )

            else:
                x = layer(x,
                          freqs=freqs,
                          attention_mask=causal_mask, )

        x = self.model.norm(x)

        return self.lm_head(x)

    def training_step(self, batch: dict, batch_idx):
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        logits = self.forward(input_ids, attention_mask)

        loss = self.loss_fn(logits[..., :-1, :].flatten(0, 1), labels[..., 1:].flatten(), )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ppl", torch.exp(loss), on_step=True, on_epoch=True, prog_bar=True)

        lrs = [param_group['lr'] for param_group in self.trainer.optimizers[0].param_groups]
        for i, lr in enumerate(lrs):
            self.log(f"lr_{i}", lr, on_step=True, on_epoch=True, )

        return loss

    def validation_step(self, batch: dict, batch_idx):
        input_ids = batch.get("input_ids")
        audio = batch.get("audio")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        logits = self(input_ids, attention_mask)

        loss = self.loss_fn(logits[..., :-1, :].flatten(0, 1), labels[..., 1:].flatten(), )

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_ppl", torch.exp(loss), on_step=True, on_epoch=True, prog_bar=True)

        return loss
