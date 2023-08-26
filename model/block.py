from typing import Optional, Tuple

import tensorflow as tf

from model.attention_layer import Attention
from model.config import ModelConfig
from model.feed_forward import FeedForward
from model.norm import RMSNorm


class TransformerBlock(tf.keras.layers.Layer):
    """
    A transformer block.

    Args:
        config (ModelConfig): The model configuration class.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.attention = Attention(config=config, name='attention')

        self.feed_forward = FeedForward(config=config, name='ffn')
        self.input_layernorm = RMSNorm(eps=config.rms_norm_eps, name='input_norm')
        self.post_attention_layernorm = RMSNorm(eps=config.rms_norm_eps, name='post_attention_norm')

    def call(self,
             hidden_states: tf.Tensor,
             attention_mask: Optional[tf.Tensor] = None,
             position_ids: Optional[tf.Tensor] = None,
             past_key_value: Optional[Tuple[tf.Tensor]] = None,
             output_attentions: Optional[bool] = False,
             use_cache: Optional[bool] = False,
             *args, **kwargs):
        """
        Passes the inputs through the transformer block.

        :param hidden_states: The input tensor of shape (batch_size, seq_len, dim).
        :param attention_mask: The attention mask tensor of shape (batch_size, 1, seq_len, seq_len).
        :param position_ids: The position ids tensor of shape (batch_size, seq_len).
        :param output_attentions: Whether to output the attention weights. If set to `True`, `past_key_values` key value
         states are returned and can be used to speed up decoding
        :param use_cache: Whether the model should use the past last key value states.
        :param past_key_value: Tuple of tensors containing cached key and value states of the attention blocks. Can be
            used to speed up decoding.
        :param kwargs: Additional keyword arguments.

        :return: The output tensor of shape (batch_size, seq_len, dim).
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
