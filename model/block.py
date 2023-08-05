from typing import Optional

import tensorflow as tf

from model.attention_layer import Attention
from model.feed_forward import FeedForward
from model.norm import RMSNorm


class TransformerBlock(tf.keras.layers.Layer):
    """
    A transformer block.

    Args:
        dim (int): The number of features in the input tensor.
        n_heads (int): The number of attention heads.
        hidden_dim (int): The number of features in the hidden layer. Default is None, which sets the hidden dimension
                            to 4 * dim. This is the same as the original implementation.
        max_batch_size (int): The maximum batch size.
        max_seq_len (int): The maximum sequence length.
        multiple_of (int): The dimension of the feed forward layer must be a multiple of this value.
        ffn_dim_multiplier (Optional[float]): The dimension of the feed forward layer will be multiplied by this value.
                                                Default is None, which sets the multiplier to 4.0.
        norm_eps (float): A small value used for numerical stability when normalizing.
                            Default is 1e-5.
    """

    def __init__(self, dim: int, n_heads: int, hidden_dim: Optional[int],
                 max_batch_size: int, max_seq_len: int,
                 multiple_of: int, ffn_dim_multiplier: Optional[float], norm_eps: float, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.dim = dim
        self.n_heads = n_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps

        self.head_dim = dim // n_heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else 4 * dim

        self.attention = Attention(n_heads=n_heads, dim=dim,
                                   max_batch_size=max_batch_size, max_seq_len=max_seq_len,
                                   **kwargs)

        self.feed_forward = FeedForward(dim=dim, hidden_dim=4*dim,
                                        multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier, **kwargs)
        self.attention_norm = RMSNorm(eps=norm_eps)
        self.ffn_norm = RMSNorm(eps=norm_eps)

    def call(self, x: tf.Tensor, freqs_cis: tf.Tensor, mask: Optional[tf.Tensor], **kwargs):
        """
        Passes the inputs through the transformer block.

        :param x: The input tensor of shape (batch_size, seq_len, dim).
        :param freqs_cis: The frequency tensor of shape (batch_size, seq_len, dim).
        :param mask: The attention mask of shape (batch_size, seq_len, seq_len).
        :param kwargs: Additional keyword arguments.

        :return: The output tensor of shape (batch_size, seq_len, dim).
        """
        h = x + self.attention(x=self.attention_norm(x), freqs_cis=freqs_cis, mask=mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
