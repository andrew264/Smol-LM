from typing import Optional

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
             *args, **kwargs):
        """
        Passes the inputs through the transformer block.

        :param hidden_states: The input tensor of shape (batch_size, seq_len, dim).
        :param attention_mask: The attention mask tensor of shape (batch_size, 1, seq_len, seq_len).
        :param kwargs: Additional keyword arguments.

        :return: The output tensor of shape (batch_size, seq_len, dim).
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
