from functools import partial
from typing import Optional

import bitsandbytes as bnb
import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import get_cosine_schedule_with_warmup

from model.modalities.audio_head import AudioHead
from .block import Block
from .config import ModelConfig
from .norm import get_rmsnorm_class
from .utils import (LINEAR, merge_audio_features, get_optimizer_grouped_parameters,
                    get_lora_plus_optimizer_group)

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss

    CrossEntropyLoss = partial(CrossEntropyLoss, inplace_backward=True)
except ImportError:
    from torch.nn import CrossEntropyLoss


class SmolLMLit(L.LightningModule):
    def __init__(self, config: ModelConfig, use_lora_opt_grp: bool = False, use_scheduler: bool = False):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = config.tie_word_embeddings
        self.checkpointing_layers = config.checkpointing_layers
        self.max_length = config.max_position_embeddings
        self.use_lora_opt_grp = use_lora_opt_grp
        self.use_scheduler = use_scheduler
        self.model = nn.ModuleDict(dict(
            embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            layers=nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers)),
            norm=get_rmsnorm_class()(config.hidden_size, eps=config.rms_norm_eps),
        ))
        if config.has_audio:
            self.audio_head = AudioHead(config)

        self.lm_head: Optional[LINEAR] = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        self.loss_fn = CrossEntropyLoss(ignore_index=-100)

        self.apply(self._init_weights)  # to initialize weights

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
                                                        num_warmup_steps=30,
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

    # @torch.compile
    def forward(
            self,
            input_ids: Tensor = None,
            audio: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
    ):

        x = self.model.embed_tokens(input_ids)

        if audio is not None:
            audio_features = self.audio_head(audio)
            x, attention_mask, labels = merge_audio_features(x, attention_mask, labels, audio_features, self.max_length)

        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)

        causal_mask = self._generate_causal_mask(attention_mask, x)

        for i, layer in enumerate(self.model.layers):
            if self.training and i in self.checkpointing_layers:
                x = checkpoint(layer,
                               x, causal_mask, position_ids,
                               use_reentrant=False, )

            else:
                x = layer(x,
                          attention_mask=causal_mask,
                          position_ids=position_ids, )

        x = self.model.norm(x)

        logits = self.lm_head(x)

        return logits, labels

    def training_step(self, batch: dict, batch_idx):
        input_ids = batch.get("input_ids")
        audio = batch.get("audio")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        logits, labels2 = self.forward(input_ids, audio, attention_mask, labels)
        if labels2 is not None:
            labels = labels2
        loss = self.loss_fn(logits[..., :-1, :].flatten(0, 1), labels[..., 1:].flatten(), )

        self.log("train_loss", loss, on_step=True, on_epoch=True, )
        self.log("train_ppl", torch.exp(loss), on_step=True, on_epoch=True, )

        lrs = [param_group['lr'] for param_group in self.trainer.optimizers[0].param_groups]
        for i, lr in enumerate(lrs):
            self.log(f"lr_{i}", lr, on_step=True, on_epoch=True, )

        return loss

    def validation_step(self, batch: dict, batch_idx):
        input_ids = batch.get("input_ids")
        audio = batch.get("audio")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        logits, _ = self(input_ids, audio, attention_mask, labels)

        loss = self.loss_fn(logits[..., :-1, :].flatten(0, 1), labels[..., 1:].flatten(), )

        self.log("val_loss", loss, on_step=True, on_epoch=True, )
        self.log("val_ppl", torch.exp(loss), on_step=True, on_epoch=True, )

        return loss
