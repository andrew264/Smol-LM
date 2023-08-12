from typing import Optional, Tuple

import tensorflow as tf

from model.utils import shape_list


class Attention(tf.keras.layers.Layer):
    """
    Multi-head self-attention layer with rotary positional embeddings.

    :param n_heads: The number of attention heads.
    :param dim: The dimension of the model.
    :param max_batch_size: The maximum batch size.
    :param max_seq_len: The maximum sequence length.
    """

    def __init__(self, n_heads: int, dim: int, max_batch_size: int, max_seq_len: int, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.query = tf.keras.layers.Dense(dim, use_bias=False)
        self.key = tf.keras.layers.Dense(dim, use_bias=False)
        self.value = tf.keras.layers.Dense(dim, use_bias=False)
        self.combine_heads = tf.keras.layers.Dense(dim, use_bias=False)

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
    def reshape_for_broadcast(freqs_cis: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Reshapes the tensor of precomputed rotary positional embeddings to match the shape of the input tensor.

        Args:
            freqs_cis (tf.Tensor): Precomputed rotary positional embeddings of shape (sequence_length, num_features).
            x (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_heads, num_features).

        Returns:
            tf.Tensor: The reshaped tensor of precomputed rotary positional embeddings.
                          The resulting tensor will have shape (batch_size, sequence_length, num_heads, num_features).
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # [sequence_length, num_features]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return tf.reshape(freqs_cis, shape, name='freqs_cis_reshaped')

    def apply_rotary_emb(self, xq: tf.Tensor, xk: tf.Tensor, freqs_cis: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies rotary positional embeddings to the input tensors.

        Args:
            xq (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_heads, num_features).
                            The tensor to which rotary embeddings will be applied.
            xk (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_heads, num_features).
                            The tensor to which rotary embeddings will be applied.
            freqs_cis (tf.Tensor): Precomputed rotary positional embeddings.
                                   It should have the shape (sequence_length, num_features) and complex data type.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing two tensors, xq_out and xk_out, both of shape
                                        (batch_size, sequence_length, num_features).
                                        The input tensors with rotary positional embeddings applied.
        """
        dtype = xq.dtype
        xq = tf.cast(xq, dtype=tf.float32, name='xq_cast')
        xk = tf.cast(xk, dtype=tf.float32, name='xk_cast')

        xq_complex = tf.complex(xq[..., ::2], xq[..., 1::2], name='xq_complex')
        xk_complex = tf.complex(xk[..., ::2], xk[..., 1::2], name='xk_complex')

        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_complex)

        xq_out = xq_complex * freqs_cis
        xk_out = xk_complex * freqs_cis

        xq_out = tf.stack([tf.math.real(xq_out), tf.math.imag(xq_out)], axis=-1)
        xk_out = tf.stack([tf.math.real(xk_out), tf.math.imag(xk_out)], axis=-1)

        xk_out = tf.reshape(xk_out, shape=xk.shape)
        xq_out = tf.reshape(xq_out, shape=xq.shape)

        return tf.cast(xq_out, dtype=dtype, name='xq_out_cast'), tf.cast(xk_out, dtype=dtype, name='xk_out_cast')

    @staticmethod
    def scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
                                     mask: Optional[tf.Tensor] = None,) -> tf.Tensor:
        """
        Applies scaled dot product attention to the input tensors.

        Args:
            q (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_heads, head_dim).
                            The query tensor.
            k (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_heads, head_dim).
                            The key tensor.
            v (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_heads, head_dim).
                            The value tensor.
            mask (Optional[tf.Tensor]): Input tensor of shape (1, 1, sequence_length, sequence_length).
                                            The attention mask.

        Returns:
            tf.Tensor: The output tensor of shape (batch_size, sequence_length, num_heads, head_dim).
        """
        scale = 1 / shape_list(q)[-1] ** 0.5
        seq_len = shape_list(q)[2]
        q = tf.multiply(q, scale, name='q_scaled')
        attn = tf.matmul(q, k, transpose_b=True,)  # [batch_size, n_heads, seq_len, seq_len]
        if mask is not None:
            attn = attn + mask[:, :, :seq_len, :seq_len]
        attn = tf.nn.softmax(attn, axis=-1, name="attention_scores")
        out = tf.matmul(attn, v, name='attention_output')
        return tf.transpose(out, perm=[0, 2, 1, 3], name='attention_output_transposed')

    def call(self, x: tf.Tensor, freqs_cis: tf.Tensor,
             mask: Optional[tf.Tensor] = None, **kwargs):
        """
        Applies multi-head self-attention with rotary positional embeddings to the input tensor.
        :param x: Input tensor of shape (batch_size, sequence_length, num_features).
        :param freqs_cis: Precomputed rotary positional embeddings.
        :param mask: Optional mask tensor of shape (batch_size, sequence_length).
        :return: The output tensor of shape (batch_size, sequence_length, num_features).
        """

        batch_size, seq_len, dim = shape_list(x)

        xq, xk, xv = self.query(x), self.key(x), self.value(x)  # [batch_size, seq_len, dim]

        # [batch_size, seq_len, n_heads, head_dim]
        xq = tf.reshape(xq, [batch_size, seq_len, self.n_heads, self.head_dim])
        xk = tf.reshape(xk, [batch_size, seq_len, self.n_heads, self.head_dim])
        xv = tf.reshape(xv, [batch_size, seq_len, self.n_heads, self.head_dim])
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)  # [batch_size, seq_len, n_heads, head_dim]

        xk = self.repeat_kv(xk, 1)  # [batch_size, seq_len, n_heads, head_dim]
        xv = self.repeat_kv(xv, 1)

        xq = tf.transpose(xq, perm=[0, 2, 1, 3], name='query_transpose')  # [batch_size, n_heads, seq_len, head_dim]
        xk = tf.transpose(xk, perm=[0, 2, 1, 3], name='key_transpose')
        xv = tf.transpose(xv, perm=[0, 2, 1, 3], name='value_transpose')

        output = self.scaled_dot_product_attention(xq, xk, xv, mask)

        bsz, seq_len, n_heads, head_dim = shape_list(output)
        output = tf.reshape(output, [bsz, seq_len, -1])

        output = self.combine_heads(output)  # [batch_size, seq_len, dim]

        return output
