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

        self.token_emb = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.hidden_size,
                                                   dtype=tf.float32, name='token_emb')
        self.layers = [
            TransformerBlock(config=config, name=f"tf_block_{i}")
            for i in range(config.num_hidden_layers)
        ]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=config.rms_norm_eps,
                                                       dtype=self.dtype_policy.compute_dtype, name='norm')
        # self.output_layer = SharedOutput(embedding_layer=self.token_emb)

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

    def output_projection(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the output logits of the transformer.
        """
        return tf.matmul(tf.cast(x, dtype=self.token_emb.dtype), self.token_emb.embeddings, transpose_b=True,
                         name="output_weights")

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

        input_embeds = self.token_emb(input_ids)

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
