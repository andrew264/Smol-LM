from typing import Optional, List, Tuple

import tensorflow as tf

from model.block import TransformerBlock
from model.config import ModelConfig
from model.norm import RMSNorm
from model.output_layer import SharedOutput
from model.utils import shape_list


class Transformer(tf.keras.layers.Layer):
    """
    Transformer model.

    Attributes:
        config (ModelConfig): The model configuration class.

    """

    def __init__(self, config: ModelConfig, **kwargs):
        super(Transformer, self).__init__(**kwargs, name="transformer")
        self.config = config

        self.token_emb = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.hidden_size)
        self.layers = [
            TransformerBlock(config=config, name=f"tf_block_{i}")
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(eps=config.rms_norm_eps, name='final_norm')
        self.output_layer = SharedOutput(embedding_layer=self.token_emb)

    @staticmethod
    def create_mask(seq_len: int) -> tf.Tensor:
        """
        Creates a mask to be used for the attention layer.
        :param seq_len: (int) The length of the sequence.
        :return: (tf.Tensor) The mask of shape (1, 1, seq_len, seq_len).
        """
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, 1, seq_len, seq_len), dtype=tf.bool), -1, 0,
            name="mask"
        )

    def get_embedding(self) -> tf.Tensor:
        """
        Returns the token embedding layer.
        :return: (tf.Tensor) The token embedding layer.
        """
        return self.token_emb

    def update_output_weights(self):
        """
        Updates the output layer weights to be the same as the token embedding layer.
        """
        self.output_layer.update_weights(self.token_emb)

    def _make_causal_mask(self, input_shape, dtype, past_key_values_length):
        bsz, tgt_len = input_shape
        mask = tf.fill((tgt_len, tgt_len), tf.cast(dtype.min, dtype))
        mask_cond = tf.range(mask.shape[-1])
        mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (mask.shape[-1], 1)), tf.cast(0, dtype), mask)

        if past_key_values_length > 0:
            pad = tf.zeros((tgt_len, past_key_values_length), dtype=dtype)
            mask = tf.concat([pad, mask], axis=-1)

        return tf.tile(mask[tf.newaxis, tf.newaxis, :, :], [bsz, 1, 1, 1])

    def _expand_mask(self, mask, dtype, tgt_len=None):
        bsz, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = tf.tile(mask[:, tf.newaxis, tf.newaxis, :], [1, 1, tgt_len, 1])
        inverted_mask = 1.0 - tf.cast(expanded_mask, dtype=dtype)

        dtype_min = tf.constant(dtype.min, dtype=dtype)
        inverted_mask = tf.where(tf.cast(inverted_mask, dtype=tf.bool), dtype_min, inverted_mask)

        return inverted_mask

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                dtype=inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def call(self, input_ids: tf.Tensor = None,
             attention_mask: Optional[tf.Tensor] = None,
             position_ids: Optional[tf.Tensor] = None,
             past_key_values: Optional[List[tf.Tensor]] = None,
             use_cache: Optional[bool] = None,
             output_attentions: Optional[bool] = None,
             output_hidden_states: Optional[bool] = None,
             ) -> Tuple:
        """
        Compute the output logits of the transformer.

        :param input_ids: (tf.Tensor) The input tokens of shape (batch_size, seq_len).
        :param attention_mask: (tf.Tensor, optional) The attention mask of shape (batch_size, seq_len).
        :param position_ids: (tf.Tensor, optional) The position ids of shape (batch_size, seq_len).
        :param past_key_values: (List[tf.Tensor], optional) The past key values of shape (batch_size, seq_len, hidden_size).
        :param use_cache: (bool, optional) Whether to use the cache.
        :param output_attentions: (bool, optional) Whether to output the attentions.
        :param output_hidden_states: (bool, optional) Whether to output the hidden states.
        :return: (Tuple) The output logits, the attentions, the hidden states, and the cache.
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        shape = shape_list(input_ids)
        if len(shape) == 1:
            input_ids = tf.expand_dims(input_ids, 0)
        batch_size, seq_length = shape_list(input_ids)

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = tf.range(
                past_key_values_length, seq_length_with_past, dtype=tf.int32, name="position_ids"
            )[tf.newaxis, :]
            position_ids = tf.reshape(position_ids, shape=(-1, seq_length))
        else:
            position_ids = tf.cast(position_ids, tf.int32)
            position_ids = tf.reshape(position_ids, shape=(-1, seq_length))

        input_embeds = self.token_emb(input_ids)

        if attention_mask is None:
            attention_mask = tf.ones(shape=(batch_size, seq_length_with_past), dtype=tf.bool)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape=shape, inputs_embeds=input_embeds,
            past_key_values_length=past_key_values_length,
        )

        hidden_states = input_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        output = self.output_layer(hidden_states)

        return output, next_cache, all_hidden_states, all_self_attns
