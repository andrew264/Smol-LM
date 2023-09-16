from typing import Optional

import tensorflow as tf

from model.block import TransformerBlock
from model.config import ModelConfig


class Transformer(tf.keras.layers.Layer):
    """
    Transformer model.

    Attributes:
        config (ModelConfig): The model configuration class.

    """

    def __init__(self, config: ModelConfig, **kwargs):
        super(Transformer, self).__init__(**kwargs, name="transformer")
        self.config = config

        self.embed_tokens = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.hidden_size,
                                                      dtype=self.dtype_policy.compute_dtype, name='embed_tokens')
        self.layers = [
            TransformerBlock(config=config, name=f"tf_block_{i}")
            for i in range(config.num_hidden_layers)
        ]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=config.rms_norm_eps,
                                                       dtype=self.dtype_policy.compute_dtype, name='norm')
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = tf.keras.layers.Dense(units=config.vocab_size,
                                                 dtype=self.dtype_policy.compute_dtype,
                                                 name='lm_head')

    @staticmethod
    def create_mask(seq_len: int) -> tf.Tensor:
        """
        Creates a mask to be used for the attention layer.
        :param seq_len: (int) The length of the sequence.
        :return: (tf.Tensor) The mask of shape (1, 1, seq_len, seq_len).
        """
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, 1, seq_len, seq_len), dtype=tf.bool), -1, 0,
            name="mask"
        )

    def get_embedding(self) -> tf.Tensor:
        """
        Returns the token embedding layer.
        :return: (tf.Tensor) The token embedding layer.
        """
        return self.embed_tokens

    def set_embedding(self, weight: tf.Tensor) -> None:
        """
        Sets the token embedding layer.
        :param weight: (tf.Tensor) The new embedding layer.
        """
        self.embed_tokens.set_weights([weight])

    def set_lm_head(self, weights: tf.Tensor) -> None:
        """
        Sets the lm head layer.
        :param weights: (tf.Tensor) The new lm head layer weights and bias.
        """
        if not self.config.tie_word_embeddings:
            self.lm_head.set_weights(weights)

    def output_projection(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the output logits of the transformer.
        """
        if self.config.tie_word_embeddings:
            return tf.matmul(tf.cast(x, dtype=self.embed_tokens.dtype),
                             self.embed_tokens.embeddings,
                             transpose_b=True,
                             name="output_weights")
        else:
            return self.lm_head(x)

    def _compute_mask(self, seq_len: Optional[int] = None) -> tf.Tensor:
        return tf.experimental.numpy.triu(
            tf.ones((1, 1, seq_len, seq_len,), dtype=self.dtype_policy.compute_dtype) * float('-inf'),
            k=1)

    def call(self, input_ids: tf.Tensor = None,
             attention_mask: Optional[tf.Tensor] = None, **kwargs) -> tf.Tensor:
        """
        Compute the output logits of the transformer.

        :param input_ids: (tf.Tensor) The input tokens of shape (batch_size, seq_len).
        :param attention_mask: (tf.Tensor, optional) The attention mask of shape (batch_size, seq_len).
        :return:
        """
        if isinstance(input_ids, list):
            input_ids = tf.convert_to_tensor(input_ids)
        shape = tf.shape(input_ids)
        if len(shape) == 1:
            input_ids = tf.expand_dims(input_ids, 0)
        batch_size, seq_length = shape[0], shape[1]

        input_embeds = self.embed_tokens(input_ids)

        attention_mask = self._compute_mask(seq_length)

        hidden_states = input_embeds

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)

        output = self.output_projection(hidden_states)

        return output
