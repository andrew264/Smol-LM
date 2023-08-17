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

    def call(self, x: tf.Tensor, freqs_cis: tf.Tensor, mask: Optional[tf.Tensor], **kwargs):
        """
        Passes the inputs through the transformer block.

        :param x: The input tensor of shape (batch_size, seq_len, dim).
        :param freqs_cis: The frequency tensor of shape (batch_size, seq_len, dim).
        :param mask: The attention mask of shape (batch_size, seq_len, seq_len).
        :param kwargs: Additional keyword arguments.

        :return: The output tensor of shape (batch_size, seq_len, dim).
        """
        h = x + self.attention(x=self.input_layernorm(x), freqs_cis=freqs_cis, mask=mask)
        out = h + self.feed_forward(self.post_attention_layernorm(h))
        return out
