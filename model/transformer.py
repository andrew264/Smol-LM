from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin, Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_utils import ModuleUtilsMixin

from .audio_head import AudioHead
from .block import Block
from .config import ModelConfig, AudioConfig
from .norm import get_rms_norm_class
from .utils import load_balancing_loss_func, LINEAR

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

    def __init__(self, config: ModelConfig, enable_audio: bool = False) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.tie_word_embeddings = config.tie_word_embeddings
        self.is_moe = config.is_moe
        self.checkpointing_layers = config.checkpointing_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(Block(config, idx) for idx in range(self.num_hidden_layers))
        self.norm = get_rms_norm_class(config.use_gemma_rms_norm)(self.hidden_size, eps=config.rms_norm_eps)

        if enable_audio:
            self.audio_head = AudioHead(config)

        if not self.tie_word_embeddings:
            self.lm_head: LINEAR = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        self.apply(self._init_weights)
        self._register_load_state_dict_pre_hook(self.hf_load_hook)
        if self.config.normalize_embedding:
            self.register_buffer("normalizer",
                                 torch.tensor(self.config.hidden_size ** 0.5, dtype=torch.bfloat16),
                                 persistent=False)
        self.normalize_embedding = config.normalize_embedding
        self.logits_soft_cap = config.logits_soft_cap

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

    @staticmethod
    def hf_load_hook(state_dict: dict, prefix, *args, **kwargs):
        mappings = {
            "model.": "",
            "embed_tokens": "tok_embeddings",
            ".self_attn.": ".jonkler_block.",
            ".attention.": ".jonkler_block.",
            ".temporal_block.": ".jonkler_block.",
            ".mlp.": ".feed_forward.",
            ".mlp_block.": ".feed_forward.",
            ".block_sparse_moe.": ".feed_forward.",
            ".w1.": ".gate_proj.",
            ".w2.": ".up_proj.",
            ".w3.": ".up_proj.",
            "final_norm.": "norm.",
            ".temporal_pre_norm.": ".input_layernorm.",
            ".channel_pre_norm.": ".post_attention_layernorm.",
        }

        def get_updated_key(key: str) -> str:
            updated_key = str(key)
            for old, new in mappings.items():
                updated_key = updated_key.replace(old, new)
            return updated_key

        for k in list(state_dict.keys()):
            state_dict[get_updated_key(k)] = state_dict.pop(k)

    def resize_embeddings(self, new_num_tokens: int) -> None:
        pad_to_multiple_of = 8
        new_num_tokens = (new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        device = self.tok_embeddings.weight.device
        dtype = self.tok_embeddings.weight.dtype

        new_embeddings = nn.Embedding(new_num_tokens, self.hidden_size, padding_idx=self.config.pad_token_id)
        new_embeddings.to(device=device, dtype=dtype)

        # copy token embeddings
        num_tokens_to_copy = min(self.vocab_size, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = self.tok_embeddings.weight.data[:num_tokens_to_copy, :]

        # replace the old embeddings with the new embeddings
        self.tok_embeddings = new_embeddings

        # resize the output layer
        new_lm_head = nn.Linear(self.hidden_size, new_num_tokens, bias=False).to(device=device, dtype=dtype)
        new_lm_head.weight.data[:num_tokens_to_copy, :] = self.lm_head.weight.data[:num_tokens_to_copy, :]
        self.lm_head = new_lm_head

        self.vocab_size = new_num_tokens

    def get_optimizer_grouped_parameters(self, weight_decay) -> list[dict]:
        decay_denylist = ["tok_embeddings.weight"]
        # start with all the candidate parameters
        decay = set()
        no_decay = set()
        param_dict = {}
        for name, param in self.named_parameters():
            param_dict[name] = param
            if param.ndimension() == 1 or any(nd in name for nd in decay_denylist):
                no_decay.add(name)
            else:
                decay.add(name)

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0},
            {'params': [param_dict[pn] for pn in sorted(list(decay))], 'weight_decay': weight_decay},
        ]

        return optim_groups

    def process_modalities(self, input_embeds: Tensor, labels: Optional[Tensor], audio: Tensor):
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

        combined_features = torch.cat((audio_features, input_embeds), dim=1)

        if combined_features.shape[1] > max_length:
            combined_features = combined_features[:, -max_length:]

        if labels is not None:
            truncated_labels = labels[:, -max_length:] if labels.shape[1] > max_length else labels
        else:
            truncated_labels = None

        return combined_features, truncated_labels

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            audio: Optional[Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[CausalLMOutputWithPast, MoeCausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_length = 0
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

        if input_ids is not None:
            x = self.tok_embeddings(input_ids)
        else:
            x = inputs_embeds

        if audio is not None:
            x, labels = self.process_modalities(x, labels, audio)

        if past_length == 0:
            if cache_position is None:
                cache_position = torch.arange(x.shape[1], device=x.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
        else:
            cache_position = torch.arange(past_length, past_length + x.shape[1], device=x.device)
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, x, cache_position)

        if self.normalize_embedding:
            x = x * self.normalizer.type(x.dtype)

        all_router_logits: Tuple[Tensor] = ()
        for i, layer in enumerate(self.layers):
            if self.training and i in self.checkpointing_layers:
                layer_outputs = checkpoint(layer,
                                           x, causal_mask, position_ids, cache_position,
                                           use_reentrant=False, )

            else:
                layer_outputs = layer(x,
                                      attention_mask=causal_mask,
                                      position_ids=position_ids,
                                      cache_position=cache_position, )

            x = layer_outputs[0]
            if self.is_moe:
                router_logits = layer_outputs[1]
                all_router_logits += (router_logits,)

        x = self.norm(x)

        if not self.tie_word_embeddings:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.tok_embeddings.weight)

        if self.logits_soft_cap is not None:
            cap = self.logits_soft_cap
            logits = nn.functional.tanh(logits / cap) * cap
            logits = logits.float()

        loss = None
        aux_loss = None
        if labels is not None:
            loss = self.loss_fn(logits[..., :-1, :].flatten(0, 1), labels[..., 1:].flatten(), )
            if self.is_moe:
                aux_loss = load_balancing_loss_func(all_router_logits, self.num_experts,
                                                    top_k=2, attention_mask=attention_mask, )
                loss += self.router_aux_loss_coef * aux_loss

        if self.is_moe:
            return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                             past_key_values=past_key_values, )
        else:
            return CausalLMOutputWithPast(loss=loss, logits=logits,
                                          past_key_values=past_key_values, )

    def prepare_inputs_for_generation(
            self, input_ids: Tensor, audio: Optional[Tensor] = None, past_key_values: Optional[Cache] = None,
            attention_mask: Optional[Tensor] = None, cache_position=None,
            **kwargs
    ):
        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "cache_position": cache_position,
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
