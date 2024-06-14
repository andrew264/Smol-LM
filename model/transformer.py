from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GenerationMixin, Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ModuleUtilsMixin

from model.modalities.audio_head import AudioHead
from .block import Block
from .config import ModelConfig
from .norm import get_rmsnorm_class
from .rotary import RotaryEmbedding
from .utils import LINEAR, merge_audio_features

try:
    from functools import partial
    from flash_attn.losses.cross_entropy import CrossEntropyLoss

    CrossEntropyLoss = partial(CrossEntropyLoss, inplace_backward=True)
    print("Using CrossEntropyLoss from flash_attn.")
except ImportError:
    from torch.nn import CrossEntropyLoss


class SmolLM(nn.Module, ModuleUtilsMixin, GenerationMixin):
    main_input_name = "inputs_embeds"
    _supports_cache_class = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.tie_word_embeddings = config.tie_word_embeddings
        self.checkpointing_layers = config.checkpointing_layers
        self.max_length = config.max_position_embeddings

        self.model = nn.ModuleDict(dict(
            embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            layers=nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers)),
            norm=get_rmsnorm_class()(config.hidden_size, eps=config.rms_norm_eps),
        ))

        if config.has_audio:
            self.audio_head = AudioHead(config)

        self.rotary_emb = RotaryEmbedding(dim=config.hidden_size // config.num_attention_heads,
                                          base=config.rope_theta, )

        self.lm_head: Optional[LINEAR] = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        self.apply(self._init_weights)

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

    def resize_embeddings(self, new_num_tokens: int) -> None:
        pad_to_multiple_of = 8
        new_num_tokens = (new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        device = self.model.embed_tokens.weight.device
        dtype = self.model.embed_tokens.weight.dtype

        new_embeddings = nn.Embedding(new_num_tokens, self.hidden_size, padding_idx=self.config.pad_token_id)
        new_embeddings.to(device=device, dtype=dtype)

        # copy token embeddings
        num_tokens_to_copy = min(self.vocab_size, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = self.tok_embeddings.weight.data[:num_tokens_to_copy, :]

        # replace the old embeddings with the new embeddings
        self.model.embed_tokens = new_embeddings

        # resize the output layer
        new_lm_head = nn.Linear(self.hidden_size, new_num_tokens, bias=False).to(device=device, dtype=dtype)
        new_lm_head.weight.data[:num_tokens_to_copy, :] = self.lm_head.weight.data[:num_tokens_to_copy, :]
        self.lm_head = new_lm_head

        self.config.vocab_size = new_num_tokens

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            audio: Optional[Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Cache] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> CausalLMOutputWithPast:
        x = self.model.embed_tokens(input_ids)

        if audio is not None:
            audio_features = self.audio_head(audio)
            x, attention_mask, labels = merge_audio_features(x, attention_mask, labels, audio_features, self.max_length)

        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_length, past_length + x.shape[1], device=x.device)

        # causal_mask = self._update_causal_mask(attention_mask, x, cache_position)
        freqs = self.rotary_emb(cache_position.unsqueeze(0))

        for layer in self.model.layers:
            x = layer(x,
                      freqs=freqs,
                      attention_mask=None,
                      cache_position=cache_position, )

        x = self.model.norm(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits[..., :-1, :].flatten(0, 1), labels[..., 1:].flatten(), )

        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                      past_key_values=past_key_values, )

    def prepare_inputs_for_generation(
            self, input_ids: Tensor, audio: Optional[Tensor] = None, past_key_values: Optional[Cache] = None,
            attention_mask: Optional[Tensor] = None, cache_position=None, position_ids=None,
            **kwargs
    ):

        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            cache_position = None
            position_ids = None
            audio = None

            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, past_length:]
            else:
                input_ids = input_ids[:, -1:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -1:]

        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "cache_position": cache_position,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "audio": audio
        }

        return model_inputs

    # copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    @staticmethod
    def _update_causal_mask(attention_mask, input_tensor, cache_position):
        if attention_mask is None:
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        if attention_mask is not None and attention_mask.device.type == "cuda":
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def tie_weights(self):
        pass
