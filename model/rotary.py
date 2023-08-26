from typing import Tuple

import tensorflow as tf


class LlamaRotaryEmbedding(tf.keras.layers.Layer):
    """
    Rotary positional embedding layer.

    Attributes:
        dim (int): The dimensionality of the embeddings.
        max_position_embeddings (int): The maximum sequence length.
        base (int): The base value used for the embeddings.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.inv_freq = 1.0 / (self.base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))

        self._set_cos_sin_cache(seq_len=max_position_embeddings)
        self.cos_cached: tf.Tensor
        self.sin_cached: tf.Tensor

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = tf.range(self.max_seq_len_cached, dtype=tf.float32)

        freqs = tf.einsum("i,j->ij", t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        emb = tf.cast(emb, dtype=self.dtype_policy.compute_dtype)
        self.cos_cached = tf.cos(emb)[None, None, :, :]
        self.sin_cached = tf.sin(emb)[None, None, :, :]

    def call(self, x: tf.Tensor, seq_len=None, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev

    Attributes:
        dim (int): The dimensionality of the embeddings.
        max_position_embeddings (int): The maximum sequence length.
        base (int): The base value used for the embeddings.
        scaling_factor (float): The scaling factor used for the embeddings.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0, **kwargs):
        self.scaling_factor = scaling_factor
        super(LlamaLinearScalingRotaryEmbedding, self).__init__(dim, max_position_embeddings, base, **kwargs)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = tf.range(self.max_seq_len_cached, dtype=tf.float32)
        t = t / self.scaling_factor

        freqs = tf.einsum("i,j->ij", t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        emb = tf.cast(emb, dtype=self.dtype_policy.compute_dtype)
        self.cos_cached = tf.cos(emb)[None, None, :, :]
        self.sin_cached = tf.sin(emb)[None, None, :, :]


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla

    Attributes:
        dim (int): The dimensionality of the embeddings.
        max_position_embeddings (int): The maximum sequence length.
        base (int): The base value used for the embeddings.
        scaling_factor (float): The scaling factor used for the embeddings.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0, **kwargs):
        self.scaling_factor = scaling_factor
        super(LlamaDynamicNTKScalingRotaryEmbedding, self).__init__(dim, max_position_embeddings, base, **kwargs)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))

        t = tf.range(self.max_seq_len_cached, dtype=tf.float32)

        freqs = tf.einsum("i,j->ij", t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        emb = tf.cast(emb, dtype=self.dtype_policy.compute_dtype)
        self.cos_cached = tf.cos(emb)[None, None, ...]
        self.sin_cached = tf.sin(emb)[None, None, ...]
