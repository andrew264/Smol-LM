from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from flash_attn.ops.fused_dense import FusedDense
from flash_attn.ops.triton.layer_norm import RMSNorm
from torch import Tensor
from transformers import GenerationMixin, Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_utils import ModuleUtilsMixin

from model import ModelConfig
from model.block import TransformerBlock
from model.cache import DynamicCache


def load_balancing_loss_func(gate_logits: Tensor | Tuple, num_experts: int = None, top_k=2) -> Tensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
    """
    if not any(gate_logits):  # if gate_logits is None or empty tuple
        return 0

    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    # treat `top_k` as tokens (shape is `top_k X [batch_size X sequence_length]`)
    selected_experts = selected_experts.reshape(-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    expert_mask = torch.max(expert_mask, dim=-2).values

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-1))
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
        self.output = FusedDense(self.hidden_size, self.vocab_size, bias=False)
        if self.tie_word_embeddings:  # TODO: model does not converge if we tie the weights
            self.output.weight = self.tok_embeddings.weight

        self.loss_fn = CrossEntropyLoss(inplace_backward=True)
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
            **kwargs,
    ) -> Union[CausalLMOutputWithPast, MoeCausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_ids is not None:
            x = self.tok_embeddings(input_ids)
        else:
            x = inputs_embeds

        if attention_mask is not None:
            # prepare the mask for F.scaled_dot_product_attention
            bs, seq_len = attention_mask.shape
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(bs, seq_len),
                inputs_embeds=x,
                past_key_values_length=past_key_values.get_usable_length(seq_len) if past_key_values else 0,
                sliding_window=self.sliding_window, )

        all_router_logits = ()
        next_decoder_cache = None
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(x, attention_mask=attention_mask,
                                  past_key_value=past_key_values, )
            x = layer_outputs[0]
            if self.config.is_moe:
                router_logits = layer_outputs[1]
                all_router_logits += (router_logits,)

        x = self.norm(x)

        logits = self.output(x)
        loss = None
        if labels is not None:
            _logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(_logits.view(-1, self.vocab_size),
                                labels.view(-1), )
            if self.config.is_moe:
                aux_loss = load_balancing_loss_func(all_router_logits, self.num_experts, top_k=2)
                loss += self.router_aux_loss_coef * aux_loss

        if self.config.is_moe:
            return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                             past_key_values=next_decoder_cache,)
        else:
            return CausalLMOutputWithPast(loss=loss, logits=logits,
                                          past_key_values=next_decoder_cache,)

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, DynamicCache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                raise ValueError("past_key_values must be an instance of DynamicCache")
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
                attention_mask = attention_mask[:, past_length:]
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        model_inputs = {"input_ids": input_ids.contiguous()}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def tie_weights(self):
        pass
