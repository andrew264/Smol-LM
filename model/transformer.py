from typing import Optional, Tuple

import tensorflow as tf

from model.block import TransformerBlock
from model.config import ModelConfig
from model.output_layer import SharedOutput
from model.utils import shape_list


class Transformer(tf.keras.layers.Layer):
    """
    Transformer model.

    Attributes:
        config (ModelConfig): The model configuration class.

    """

    def __init__(self, config: ModelConfig, **kwargs):
        super(Transformer, self).__init__(**kwargs, name="transformer")
        self.config = config

        self.token_emb = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.hidden_size,
                                                   dtype=self.dtype_policy.compute_dtype, name='token_emb')
        self.layers = [
            TransformerBlock(config=config, name=f"tf_block_{i}")
            for i in range(config.num_hidden_layers)
        ]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=config.rms_norm_eps,
                                                       dtype=self.dtype_policy.compute_dtype, name='norm')
        # self.output_layer = SharedOutput(embedding_layer=self.token_emb)
        self.output_layer = tf.keras.layers.Dense(units=config.vocab_size, use_bias=False,
                                                  dtype=self.dtype_policy.compute_dtype, name='output')

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
        return self.token_emb

    def update_output_weights(self):
        """
        Updates the output layer weights to be the same as the token embedding layer.
        """
        # self.output_layer.update_weights(self.token_emb)
        pass

    def _compute_mask(self, seq_len: Optional[int] = None) -> tf.Tensor:
        return tf.experimental.numpy.triu(
            tf.ones((1, 1, seq_len, seq_len,), dtype=self.dtype_policy.compute_dtype) * float('-inf'),
            k=1)

    def call(self, input_ids: tf.Tensor = None,
             attention_mask: Optional[tf.Tensor] = None, **kwargs) -> Tuple:
        """
        Compute the output logits of the transformer.

        :param input_ids: (tf.Tensor) The input tokens of shape (batch_size, seq_len).
        :param attention_mask: (tf.Tensor, optional) The attention mask of shape (batch_size, seq_len).
        :return:
        """
        if isinstance(input_ids, list):
            input_ids = tf.convert_to_tensor(input_ids)
        shape = shape_list(input_ids)
        if len(shape) == 1:
            input_ids = tf.expand_dims(input_ids, 0)
        batch_size, seq_length = shape_list(input_ids)

        input_embeds = self.token_emb(input_ids)

        attention_mask = self._compute_mask(seq_length)

        hidden_states = input_embeds

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)

        output = self.output_layer(hidden_states)

        return output
