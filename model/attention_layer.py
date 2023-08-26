from typing import Tuple

import tensorflow as tf

from model.config import ModelConfig
from model.rotary import LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding
from model.utils import shape_list


class Attention(tf.keras.layers.Layer):
    """
    Multi-head self-attention layer with rotary positional embeddings.

    :param config: The model configuration class.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super(Attention, self).__init__(**kwargs)
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

        self.q_proj = tf.keras.layers.Dense(units=self.num_heads * self.head_dim,
                                            use_bias=False, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(units=self.num_key_value_heads * self.head_dim,
                                            use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(units=self.num_key_value_heads * self.head_dim,
                                            use_bias=False, name="v_proj")

        self.o_proj = tf.keras.layers.Dense(units=self.hidden_size,
                                            use_bias=False, name="o_proj")

        self._softmax = tf.nn.softmax

        self.rotary_emb = self._init_rope()

    def _init_rope(self) -> LlamaRotaryEmbedding:
        if self.config.rope_scaling is None:
            return LlamaRotaryEmbedding(dim=self.head_dim,
                                        max_position_embeddings=self.max_position_embeddings,
                                        name="rotary_emb")
        else:
            if self.config.rope_scaling.get("type") == "linear":
                return LlamaLinearScalingRotaryEmbedding(dim=self.head_dim,
                                                         max_position_embeddings=self.max_position_embeddings,
                                                         scaling_factor=self.config.rope_scaling.get("factor", 1.0),
                                                         name="linear_rotary_emb")
            elif self.config.rope_scaling.get("type") == "dynamic":
                return LlamaDynamicNTKScalingRotaryEmbedding(dim=self.head_dim,
                                                             max_position_embeddings=self.max_position_embeddings,
                                                             scaling_factor=self.config.rope_scaling.get("factor", 1.0),
                                                             name="dynamic_rotary_emb", )
            else:
                raise ValueError(f"Unknown rope scaling type: {self.config.rope_scaling.get('type')}")

    @staticmethod
    def repeat_kv(x: tf.Tensor, n_rep: int = 1) -> tf.Tensor:
        """
        Repeats each element of the tensor multiple times along the specified axis.

        This function is equivalent to `torch.repeat_interleave(x, dim=2, repeats=n_rep)` in PyTorch.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, sequence_length, n_kv_heads, head_dim).
                           The tensor to be repeated along the specified axis.
            n_rep (int): The number of times to repeat each element along the specified axis.

        Returns:
            tf.Tensor: The tensor with elements repeated `n_rep` times along the specified axis.
                       The resulting tensor will have shape (batch_size, sequence_length, n_kv_heads * n_rep, head_dim).
        """
        if n_rep == 1:
            return x
        bs, slen, n_kv_heads, head_dim = shape_list(x)  # [batch_size, sequence_length, n_kv_heads, head_dim]
        return tf.reshape(
            tf.tile(tf.expand_dims(x, axis=3), multiples=[1, 1, 1, n_rep, 1]),
            shape=[bs, slen, n_kv_heads * n_rep, head_dim]
        )

    @staticmethod
    def rotate_half(x: tf.Tensor) -> tf.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return tf.concat((-x2, x1), axis=-1)

    def apply_rotary_pos_emb(self, q: tf.Tensor, k: tf.Tensor, cos: tf.Tensor, sin: tf.Tensor, position_ids: tf.Tensor,
                             ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies rotary positional embeddings to the input tensors.

        Args:
            q (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_heads, head_dim).
                            The query tensor.
            k (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_heads, head_dim).
                            The key tensor.
            cos (tf.Tensor): Input tensor of shape (1, 1, sequence_length, head_dim).
                            The cosine positional embeddings.
            sin (tf.Tensor): Input tensor of shape (1, 1, sequence_length, head_dim).
                            The sine positional embeddings.
            position_ids (tf.Tensor): Input tensor of shape (batch_size, sequence_length).
                            The position ids.

        """
        with tf.name_scope('apply_rotary_emb'):
            cos = cos[0, 0]  # [seq_len, dim]
            sin = sin[0, 0]  # [seq_len, dim]
            cos = tf.expand_dims(tf.gather(cos, position_ids), axis=1)  # [bs, 1, seq_len, dim]
            sin = tf.expand_dims(tf.gather(sin, position_ids), axis=1)  # [bs, 1, seq_len, dim]
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
            return q_embed, k_embed

    def call(self, hidden_states,
             attention_mask=None,
             position_ids=None,
             past_key_value=None,
             output_attentions=False,
             use_cache=False,
             **kwargs):
        """
        Applies multi-head self-attention with rotary positional embeddings to the input tensor.
        :param hidden_states: Input tensor of shape (batch_size, sequence_length, num_features).
        :param attention_mask: Input tensor of shape (batch_size, 1, sequence_length, sequence_length).
            The attention mask.
        :param position_ids: Input tensor of shape (batch_size, sequence_length).
        :param past_key_value: Tuple of tensors containing cached key and value states of the attention blocks. Can be
            used to speed up decoding.
        :param output_attentions: Whether to output the attention weights. If set to `True`, `past_key_values` key value
            states are returned and can be used to speed up decoding
        :param use_cache: Whether the model should use the past last key value states.
        :param kwargs: Additional keyword arguments.

        :return: The output tensor of shape (batch_size, sequence_length, num_features).
        """

        bsz, q_len, _ = shape_list(hidden_states)

        query_states = self.q_proj(hidden_states)  # [batch_size, seq_len, dim]
        key_states = self.k_proj(hidden_states)  # [batch_size, seq_len, dim]
        value_states = self.v_proj(hidden_states)  # [batch_size, seq_len, dim]

        # [batch_size, seq_len, n_heads, head_dim]
        query_states = tf.reshape(query_states, [bsz, q_len, self.num_heads, self.head_dim])
        key_states = tf.reshape(key_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states = tf.reshape(value_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        query_states = tf.transpose(query_states, perm=[0, 2, 1, 3],
                                    name='query_transpose')  # [batch_size, n_heads, seq_len, head_dim]
        key_states = tf.transpose(key_states, perm=[0, 2, 1, 3], name='key_transpose')
        value_states = tf.transpose(value_states, perm=[0, 2, 1, 3], name='value_transpose')

        kv_seq_len = shape_list(key_states)[-2]
        if past_key_value is not None:
            kv_seq_len += shape_list(past_key_value[0])[-2]

        cos, sin = self.rotary_emb(x=value_states, seq_len=kv_seq_len)
        query_states, key_states = self.apply_rotary_pos_emb(q=query_states, k=key_states, cos=cos, sin=sin,
                                                             position_ids=position_ids)

        if past_key_value is not None:
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        with tf.name_scope('scaled_dot_product_attention'):
            attn_weights = tf.matmul(query_states, key_states, transpose_b=True) / tf.math.sqrt(
                tf.cast(self.head_dim, dtype=self.dtype_policy.compute_dtype)
            )
            if shape_list(attn_weights) != [bsz, self.num_heads, q_len, kv_seq_len]:
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is "
                    f"{shape_list(attn_weights)}"
                )
            if attention_mask is not None:
                if shape_list(attention_mask) != [bsz, 1, q_len, kv_seq_len]:
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is "
                        f"{shape_list(attention_mask)}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = tf.maximum(attn_weights, attn_weights.dtype.min)

            attn_weights = self._softmax(attn_weights, axis=-1, name='attention_softmax')
            attn_weights = tf.cast(attn_weights, dtype=self.dtype_policy.compute_dtype)
            attn_output = tf.matmul(attn_weights, value_states)

            if shape_list(attn_output) != [bsz, self.num_heads, q_len, self.head_dim]:
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {shape_list(attn_output)}"
                )
            attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
            attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
