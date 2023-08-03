import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    """
    Root Mean Square layer normalization (RMSNorm).

    Args:
        eps (float): A small value used for numerical stability when normalizing.
                        Default is 1e-6.
    """

    def __init__(self, eps=1e-6, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.eps = eps

    def get_config(self):
        config = super().get_config()
        config.update({
            "eps": self.eps,
        })
        return config

    def _norm(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)

    def call(self, inputs, **kwargs):
        """
        Passes the inputs through the RMSNorm layer.
        :param inputs: The input tensor of shape (batch_size, seq_len, dim).
        :return: The output tensor of shape (batch_size, seq_len, dim).
        """
        return self._norm(inputs)
