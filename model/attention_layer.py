import tensorflow as tf

from model.config import ModelConfig
from model.rotary import RotaryEmbedding


class Attention(tf.keras.layers.Layer):
    """
    Multi-head self-attention layer with rotary positional embeddings.

    :param config: The model configuration class.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.o_proj = None
        self.v_proj = None
        self.k_proj = None
        self.q_proj = None
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self._softmax = tf.nn.softmax

        if self.config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim, name="rotary_emb")
        else:
            if self.config.rope_scaling.get("type") == "linear":
                self.rotary_emb = RotaryEmbedding(dim=self.head_dim,
                                                  scaling_factor=self.config.rope_scaling.get("factor", 1.0),
                                                  name="linear_rotary_emb")
            else:
                raise ValueError(f"Unknown rope scaling type: {self.config.rope_scaling.get('type')}")

    def build(self, inputs_shape):
        self.q_proj = tf.keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.head_dim),
            dtype=self.dtype_policy.compute_dtype,
            name="query_proj",
        )
        self.q_proj.build(inputs_shape)

        self.k_proj = tf.keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            dtype=self.dtype_policy.compute_dtype,
            name="key_proj",
        )
        self.k_proj.build(inputs_shape)

        self.v_proj = tf.keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            dtype=self.dtype_policy.compute_dtype,
            name="value_proj",
        )
        self.v_proj.build(inputs_shape)

        self.o_proj = tf.keras.layers.EinsumDense(
            equation="abc,cd->abd",
            output_shape=(None, self.hidden_size),
            dtype=self.dtype_policy.compute_dtype,
            name="attention_output",
        )
        self.o_proj.build(inputs_shape)

        self.built = True

    def call(self, hidden_states,
             attention_mask=None,
             **kwargs):
        """
        Applies multi-head self-attention with rotary positional embeddings to the input tensor.
        :param hidden_states: Input tensor of shape (batch_size, sequence_length, num_features).
        :param attention_mask: Input tensor of shape (batch_size, 1, sequence_length, sequence_length).
            The attention mask.
        :param kwargs: Additional keyword arguments.

        :return: The output tensor of shape (batch_size, sequence_length, num_features).
        """

        shape = tf.shape(hidden_states)
        bsz, q_len = shape[0], shape[1]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.rotary_emb(query_states)
        key_states = self.rotary_emb(key_states)

        key_states = tf.tile(key_states, [1, 1, self.num_key_value_groups, 1])
        value_states = tf.tile(value_states, [1, 1, self.num_key_value_groups, 1])

        with tf.name_scope('scaled_dot_product_attention'):
            attn_scores = tf.einsum("aecd,abcd->acbe", key_states, query_states)
            norm_factor = tf.math.sqrt(tf.cast(self.head_dim, dtype=attn_scores.dtype))
            attn_scores /= norm_factor

            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
                attn_scores = tf.maximum(attn_scores, attn_scores.dtype.min)

            attn_scores = self._softmax(attn_scores, axis=-1, name='attention_softmax')
            attn_output = tf.einsum("acbe,aecd->abcd", attn_scores, value_states)

            attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.o_proj(attn_output)

        return attn_output
