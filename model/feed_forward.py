import tensorflow as tf

from model.config import ModelConfig


class FeedForward(tf.keras.layers.Layer):
    """
    A feed forward layer.

    Args:
        config (ModelConfig): The model configuration class.

    """

    def __init__(self, config: ModelConfig,
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.config = config
        hidden_size = config.hidden_size
        intermediate_size = int(2 * config.intermediate_size / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            intermediate_size = int(config.ffn_dim_multiplier * intermediate_size)
        intermediate_size = config.multiple_of * ((intermediate_size + config.multiple_of - 1) // config.multiple_of)

        self.gate_proj = tf.keras.layers.Dense(units=intermediate_size,
                                               use_bias=False,
                                               kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                   stddev=config.initializer_range),
                                               name='gate_proj')
        self.up_proj = tf.keras.layers.Dense(units=intermediate_size,
                                             use_bias=False,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                 stddev=config.initializer_range),
                                             name='up_proj')
        self.down_proj = tf.keras.layers.Dense(units=hidden_size,
                                               use_bias=False,
                                               kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                   stddev=config.initializer_range),
                                               name='down_proj')

        self.act_fn = tf.keras.activations.get(config.hidden_act)

    def call(self, x, **kwargs):
        """
        Passes the inputs through the feed forward layer.
        :param x: (tf.Tensor): The input tensor of shape (batch_size, seq_len, dim).
        :return: (tf.Tensor): The output tensor of shape (batch_size, seq_len, dim).
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
