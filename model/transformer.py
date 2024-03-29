from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin, Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_utils import ModuleUtilsMixin

from model import ModelConfig
from model.block import TransformerBlock
from model.cache import DynamicCache
from model.norm import RMSNorm


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
def load_balancing_loss_func(gate_logits: Tuple[torch.Tensor], num_experts: int, top_k: int = 2,
                             attention_mask: Optional[torch.Tensor] = None) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *required*):
            Number of experts
        top_k (`int`, *optional*):
            Number of experts to route to.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    assert isinstance(gate_logits, tuple), "gate_logits should be a tuple of tensors"
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Transformer(nn.Module, ModuleUtilsMixin, GenerationMixin):
    main_input_name = "inputs_embeds"
    _supports_cache_class = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.sliding_window = config.sliding_window
        self.tie_word_embeddings = config.tie_word_embeddings

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(TransformerBlock(config, idx) for idx in range(self.num_hidden_layers))
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        if self.tie_word_embeddings:  # TODO: model does not converge if we tie the weights
            self.lm_head.weight = self.tok_embeddings.weight

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
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

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[CausalLMOutputWithPast, MoeCausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_ids is not None:
            x = self.tok_embeddings(input_ids)
        else:
            x = inputs_embeds

        past_seen_tokens = 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + x.shape[1], device=x.device
            )
        position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, x, cache_position)

        all_router_logits = ()
        for i, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing == 'full' and self.training:
                layer_outputs = checkpoint(layer,
                                           x, causal_mask, position_ids, past_key_values,
                                           use_reentrant=False, )

            else:
                layer_outputs = layer(x, attention_mask=causal_mask,
                                      position_ids=position_ids,
                                      past_key_value=past_key_values, )
            x = layer_outputs[0]
            if self.config.is_moe:
                router_logits = layer_outputs[1]
                all_router_logits += (router_logits,)

        x = self.norm(x)

        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            _logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            _logits = _logits.transpose(1, 2)
            loss = self.loss_fn(_logits, labels, )
            if self.config.is_moe:
                aux_loss = load_balancing_loss_func(all_router_logits, self.num_experts,
                                                    top_k=2, attention_mask=attention_mask, )
                loss += self.router_aux_loss_coef * aux_loss

        if self.config.is_moe:
            return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                             past_key_values=past_key_values, )
        else:
            return CausalLMOutputWithPast(loss=loss, logits=logits,
                                          past_key_values=past_key_values, )

    def prepare_inputs_for_generation(
            self, input_ids: Tensor, past_key_values: Optional[DynamicCache] = None,
            attention_mask: Optional[Tensor] = None, cache_position=None,
            **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
                attention_mask = attention_mask[:, past_length:]
        else:
            past_length = 0

        model_inputs = {"input_ids": input_ids.contiguous()}
        input_length = input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {"cache_position": cache_position, "past_key_values": past_key_values, "attention_mask": attention_mask, }
        )
        return model_inputs

    # copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    @staticmethod
    def _update_causal_mask(attention_mask, input_tensor, cache_position):

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
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

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
