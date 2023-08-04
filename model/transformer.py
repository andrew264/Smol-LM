from typing import Optional

import tensorflow as tf

from model.block import TransformerBlock
from model.norm import RMSNorm
from model.utils import shape_list


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
        dim (int): The dimension of the model.
        n_layers (int): The number of layers in the model.
        n_heads (int): The number of attention heads.
        vocab_size (int): The size of the vocabulary.
        max_seq_len (int): The maximum sequence length.
        max_batch_size (int): The maximum batch size.
        multiple_of (int): The dimension of the feed forward layer must be a multiple of this value.
        ffn_dim_multiplier (Optional[float]): The dimension of the feed forward layer will be multiplied by this value.
                                                Default is None, which sets the multiplier to 4.0.
        norm_eps (float): A small value used for numerical stability when normalizing.
                            Default is 1e-5.

    """

    def __init__(self, dim: int, n_layers: int, n_heads: int,
                 vocab_size: int, max_seq_len: int, max_batch_size: int,
                 multiple_of: int, ffn_dim_multiplier: Optional[float],
                 norm_eps: float, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=dim)
        self.layers = [
            TransformerBlock(dim=dim, n_heads=n_heads, multiple_of=multiple_of,
                             max_batch_size=max_batch_size, max_seq_len=max_seq_len,
                             ffn_dim_multiplier=ffn_dim_multiplier, norm_eps=norm_eps)
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(eps=norm_eps)
        self.output_layer = tf.keras.layers.Dense(vocab_size, use_bias=False, name="output_layer", dtype=tf.float32)

        self.freqs_cis = precompute_freqs_cis(dim=dim // n_heads, end=max_seq_len * 2)

    def call(self, tokens: tf.Tensor, **kwargs):
        """
        Compute the output logits of the transformer.

        :param tokens: (tf.Tensor) The input tokens of shape (batch_size, seq_len).
        :return: (tf.Tensor) The output logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = shape_list(tokens)

        h = self.token_emb(tokens)
        freqs_cis = self.freqs_cis[:seq_len]

        mask = tf.fill((1, 1, seq_len, seq_len), float('-inf'))
        # Set upper triangle to float("-inf")
        mask = tf.linalg.band_part(mask, 0, -1)
        # Set diagonal to 0
        diag = tf.fill((1, 1, seq_len), 0.0)
        mask = tf.cast(tf.linalg.set_diag(mask, diag), dtype=self.dtype_policy.compute_dtype)

        for layer in self.layers:
            h = layer(x=h, freqs_cis=freqs_cis, mask=mask)
        h = self.norm(h)
        output = self.output_layer(h)
        return output
