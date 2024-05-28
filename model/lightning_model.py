from functools import partial
from typing import Optional

import bitsandbytes as bnb
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import get_cosine_schedule_with_warmup

from utils import get_lora_plus_optimizer_group
from .audio_head import AudioHead
from .block import Block
from .config import ModelConfig
from .norm import get_rmsnorm_class
from .utils import LINEAR, hf_load_hook

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss

    CrossEntropyLoss = partial(CrossEntropyLoss, inplace_backward=True)
except ImportError:
    from torch.nn import CrossEntropyLoss


class SmolLMLightning(L.LightningModule):
    def __init__(self, config: ModelConfig, use_lora_opt_grp: bool = False, use_scheduler: bool = False):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = config.tie_word_embeddings
        self.checkpointing_layers = config.checkpointing_layers
        self.use_lora_opt_grp = use_lora_opt_grp
        self.use_scheduler = use_scheduler
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.norm = get_rmsnorm_class()(config.hidden_size, eps=config.rms_norm_eps)

        if config.has_audio:
            self.audio_head = AudioHead(config)

        self.lm_head: Optional[LINEAR] = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.loss_fn = CrossEntropyLoss(ignore_index=-100)

        self.apply(self._init_weights)  # to initialize weights
        self._register_load_state_dict_pre_hook(hf_load_hook)  # to load from HF models

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
                params=self.parameters(),
                lr=lr,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
        if self.use_scheduler:
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=50,
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

    def process_modalities(self, input_embeds: Tensor,
                           attention_mask: Optional[Tensor],
                           labels: Optional[Tensor],
                           audio: Tensor):
        device = input_embeds.device
        max_length = self.config.max_position_embeddings

        # Compute audio features
        audio_features = checkpoint(self.audio_head, audio, use_reentrant=False)
        # feature_length = audio_features.shape[1]

        if labels is not None:
            label_pad = torch.full(
                (audio_features.shape[0], audio_features.shape[1]),
                -100,
                dtype=torch.long,
                device=device
            )
            labels = torch.cat((label_pad, labels), dim=1)

        if attention_mask is not None:
            attention_pad = torch.ones(
                (audio_features.shape[0], audio_features.shape[1]),
                dtype=torch.long,
                device=device
            )
            attention_mask = torch.cat((attention_pad, attention_mask), dim=1)

        combined_features = torch.cat((audio_features, input_embeds), dim=1)

        if combined_features.shape[1] > max_length:
            combined_features = combined_features[:, -max_length:]

        if labels is not None:
            truncated_labels = labels[:, -max_length:] if labels.shape[1] > max_length else labels
        else:
            truncated_labels = None

        return combined_features, attention_mask, truncated_labels

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
            audio: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
    ):
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)

        x = self.embed_tokens(input_ids)

        if audio is not None:
            x, attention_mask, labels = self.process_modalities(x, attention_mask, labels, audio)

        causal_mask = self._generate_causal_mask(attention_mask, x)

        for i, layer in enumerate(self.layers):
            if self.training and i in self.checkpointing_layers:
                x = checkpoint(layer,
                               x, causal_mask, position_ids,
                               use_reentrant=False, )

            else:
                x = layer(x,
                          attention_mask=causal_mask,
                          position_ids=position_ids, )

        x = self.norm(x)

        if not self.tie_word_embeddings:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        return logits

    def training_step(self, batch: dict, batch_idx):
        input_ids = batch.get("input_ids")
        audio = batch.get("audio")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        logits = self(input_ids, audio, attention_mask, labels)

        loss = self.loss_fn(logits[..., :-1, :].flatten(0, 1), labels[..., 1:].flatten(), )

        self.log("train_loss", loss, on_step=True, on_epoch=True, )
        self.log("train_ppl", torch.exp(loss), on_step=True, on_epoch=True, )
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, )

        return loss

    def validation_step(self, batch: dict, batch_idx):
        input_ids = batch.get("input_ids")
        audio = batch.get("audio")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        logits = self(input_ids, audio, attention_mask, labels)

        loss = self.loss_fn(logits[..., :-1, :].flatten(0, 1), labels[..., 1:].flatten(), )

        self.log("val_loss", loss, on_step=True, on_epoch=True, )
        self.log("val_ppl", torch.exp(loss), on_step=True, on_epoch=True, )

        return loss
