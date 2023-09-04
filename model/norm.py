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
        self.weight = None
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(input_shape[-1],),
            initializer="ones",
        )
        self.built = True

    def call(self, x, **kwargs):
        """
        Passes the inputs through the RMSNorm layer.
        :param x: The input tensor of shape (batch_size, seq_len, dim).
        :return: The output tensor of shape (batch_size, seq_len, dim).
        """
        hidden_states = x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return hidden_states * self.weight
