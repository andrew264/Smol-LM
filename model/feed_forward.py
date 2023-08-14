from typing import Optional

import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    """
    A feed forward layer.

    Args:
        dim (int): The number of features in the input tensor.
        hidden_dim (int): The number of features in the hidden layer.
        multiple_of (int): The dimension of the feed forward layer must be a multiple of this value.
        ffn_dim_multiplier (Optional[float]): The dimension of the feed forward layer will be multiplied by this value.
                                                Default is None, which sets the multiplier to 4.0.

    """

    def __init__(self, dim: int,
                 hidden_dim: int,
                 multiple_of: int,
                 ffn_dim_multiplier: Optional[float] = None,
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='ffn1', activation=tf.nn.silu)
        self.w2 = tf.keras.layers.Dense(dim, use_bias=False, name='ffn2')
        self.w3 = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='ffn3')
        # self.act = tf.nn.silu

    def call(self, x, **kwargs):
        """
        Passes the inputs through the feed forward layer.
        :param x: (tf.Tensor): The input tensor of shape (batch_size, seq_len, dim).
        :return: (tf.Tensor): The output tensor of shape (batch_size, seq_len, dim).
        """
        return self.w2(self.w1(x) * self.w3(x))
