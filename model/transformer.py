import tensorflow as tf

from model.block import TransformerBlock
from model.config import ModelConfig
from model.norm import RMSNorm
from model.output_layer import SharedOutput
from model.utils import shape_list


@tf.function
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tf.Tensor:
    """
    Precomputes rotary positional embeddings to be used with `apply_rotary_emb`.

    :param dim: (int): The dimensionality of the embeddings.
    :param end: (int): The maximum sequence length.
    :param theta: (float, optional): The theta value used for the embeddings.

    :return: (tf.Tensor): The precomputed embeddings of dtype=tf.complex64
    """
    freqs = 1.0 / (theta ** (tf.range(0, dim, 2, dtype=tf.float32) / dim))
    t = tf.range(end, dtype=tf.float32)
    freqs = tf.matmul(tf.expand_dims(t, -1), tf.expand_dims(freqs, 0))
    freqs_cis = tf.exp(tf.complex(tf.zeros_like(freqs), freqs))
    return freqs_cis


class Transformer(tf.keras.layers.Layer):
    """
    Transformer model.

    Attributes:
        config (ModelConfig): The model configuration class.

    """

    def __init__(self, config: ModelConfig, **kwargs):
        super(Transformer, self).__init__(**kwargs, name="transformer")
        self.config = config

        self.token_emb = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.hidden_size)
        self.layers = [
            TransformerBlock(config=config, name=f"tf_block_{i}")
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(eps=config.rms_norm_eps, name='final_norm')
        self.output_layer = SharedOutput(embedding_layer=self.token_emb)
        self.freqs_cis = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                              end=config.max_position_embeddings * 2)

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
        self.output_layer.update_weights(self.token_emb)

    def call(self, tokens: tf.Tensor, **kwargs):
        """
        Compute the output logits of the transformer.

        :param tokens: (tf.Tensor) The input tokens of shape (batch_size, seq_len).
        :return: (tf.Tensor) The output logits of shape (batch_size, seq_len, vocab_size).
        """
        shape = shape_list(tokens)
        if len(shape) == 1:
            tokens = tf.expand_dims(tokens, 0)
        seq_len = shape_list(tokens)[1]

        h = self.token_emb(tokens)
        freqs_cis = self.freqs_cis[:seq_len]
        mask = self.create_mask(seq_len=seq_len)

        for layer in self.layers:
            h = layer(x=h, freqs_cis=freqs_cis, mask=mask)
        h = self.norm(h)
        output = self.output_layer(h)
        return output
