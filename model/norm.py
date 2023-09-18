import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    """
    Root Mean Square layer normalization (RMSNorm).

    Args:
        eps (float): A small value used for numerical stability when normalizing.
                        Default is 1e-6.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.gamma = None
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=input_shape[-1],
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Passes the inputs through the RMSNorm layer.
        :param inputs: The input tensor of shape (batch_size, seq_len, dim).
        :return: The output tensor of shape (batch_size, seq_len, dim).
        """
        input_dtype = inputs.dtype
        if input_dtype in ("float16", "bfloat16") and self.dtype == "float32":
            # If mixed precision is used, cast inputs to float32 so that
            # this is at least as numerically stable as the fused version.
            inputs = tf.cast(inputs, tf.float32)
        variance = tf.math.reduce_variance(inputs, axis=-1, keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        outputs = inputs * inv * tf.cast(self.gamma, tf.float32)

        return tf.cast(outputs, input_dtype)
