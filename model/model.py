import re
from typing import List, Optional

import tensorflow as tf
import tensorflow_probability as tfp
import tqdm

from model.config import ModelConfig
from model.tokenizer import Tokenizer
from model.transformer import Transformer
from model.utils import GradientAccumulator, shape_list


def scatter_values_on_batch_indices(values, batch_indices):
    shape = shape_list(batch_indices)
    # broadcast batch dim to shape
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.range(shape[0]), axis=-1), shape), [1, -1])
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), shape)


class SmolLM(tf.keras.Model):
    """
    SmolLM model.

    Attributes:
        config (ModelConfig): The model configuration class.
        num_accumulation (int): The number of gradient accumulation steps.
                                 Default is 4.
    """

    def __init__(self, config: ModelConfig, num_accumulation: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.transformer = Transformer(config=config)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.perplexity_tracker = tf.keras.metrics.Mean(name="perplexity")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")
        self._gradient_accumulator = GradientAccumulator()
        self._gradient_accumulator.reset()
        self.num_accumulation = num_accumulation
        self._tokenizer: Optional[Tokenizer] = None

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("tokenizer must be an instance of Tokenizer.")
        self._tokenizer = tokenizer

    def call(self, input_ids: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Calls the model.

        :param input_ids: (tf.Tensor) The input tokens tensor.
        :param kwargs: Additional keyword arguments.
        :return: The output logits tensor.
        """
        return self.transformer(input_ids=input_ids, **kwargs)

    def load_weights(self, filepath, skip_mismatch=False, by_name=False, options=None):
        """
        Loads the model weights.
        After loading the weights, the output layer weights are set to the embedding layer weights.
        """
        super().load_weights(filepath, skip_mismatch, by_name, options)
        self.transformer.update_output_weights()

    def get_config(self):
        config = super(SmolLM, self).get_config()
        config.update({"config": self.config, "num_accumulation": self.num_accumulation})

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        model = cls(**kwargs)
        model.load_weights(path)
        return model

    @tf.function(jit_compile=True)
    def get_padded_accuracy(self, labels, logits):
        with tf.name_scope("padded_accuracy"):
            pred = tf.argmax(logits, axis=-1)
            labels = tf.cast(labels, dtype=pred.dtype)
            match = labels == pred

            mask = labels != 0
            match = match & mask
            match = tf.cast(match, dtype=tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)
            accuracy = tf.reduce_sum(match) / tf.reduce_sum(mask)
            return accuracy

    @tf.function(jit_compile=True)
    def get_perplexity(self, cross_entropy):
        with tf.name_scope("perplexity"):
            return tf.exp(cross_entropy)

    @tf.function(jit_compile=True)
    def get_loss(self, real, pred):
        with tf.name_scope("loss"):
            mask = real != 0
            loss_ = self.loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            loss_ = tf.reduce_sum(loss_, axis=-1) / tf.reduce_sum(mask, axis=-1)
            loss_ = tf.reduce_mean(loss_)
            return loss_

    @property
    def metrics(self):
        return [self.loss_tracker, self.perplexity_tracker, self.accuracy_tracker]

    @tf.function(jit_compile=True)
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)[0]
            loss = self.get_loss(y, logits)

        gradients = tape.gradient(loss, self.trainable_variables)

        if self.num_accumulation == 1:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        else:
            self._gradient_accumulator(gradients)
            if self._gradient_accumulator.step == self.num_accumulation:
                averaged_gradients = [gradient / tf.cast(self.num_accumulation, dtype=gradient.dtype) for gradient in
                                      self._gradient_accumulator.gradients]
                self.optimizer.apply_gradients(zip(averaged_gradients, self.trainable_variables))
                self._gradient_accumulator.reset()

        y_pred = tf.nn.softmax(logits)
        self.loss_tracker.update_state(loss)
        self.perplexity_tracker.update_state(self.get_perplexity(loss))
        accuracy = self.get_padded_accuracy(y, y_pred)
        self.accuracy_tracker.update_state(accuracy)

        return {m.name: m.result() for m in self.metrics} | {'lr': self.optimizer.learning_rate.value()}

    @tf.function(jit_compile=True)
    def test_step(self, data):
        x, y = data
        logits = self(x, training=False)[0]
        loss = self.get_loss(y, logits)

        self.loss_tracker.update_state(loss)
        self.perplexity_tracker.update_state(self.get_perplexity(loss))
        accuracy = self.get_padded_accuracy(y, logits)
        self.accuracy_tracker.update_state(accuracy)

        return {m.name: m.result() for m in self.metrics}

    def generate(self, idx: List[List[int]], max_gen_len: int,
                 temperature: float = 0.6, top_k: int = 0, top_p: float = 0.0, stream: bool = False
                 ) -> List[List[int]]:
        """
        Generates text from a prompt.

        From: https://github.com/huggingface/transformers/blob/fe3c8ab1af558b95f67f5fafc0c55f09fd2b09db/src/transformers/generation/tf_utils.py#L3059

        Params:
        :param idx: A list of lists of integers. Each list of integers is a prompt.
        :param max_gen_len: The maximum length of the generated text.
        :param temperature: The temperature to use when sampling from the softmax distribution.
        :param top_k: The number of top tokens to consider when sampling from the softmax distribution.
        :param top_p: The cumulative probability of top tokens to consider when sampling from the softmax distribution.
        :param stream: Whether to stream the generation or not. If True, the generation is streamed to the output.
        :return: A list of lists of integers. Each list of integers is a generated text.
        """
        if stream and self.tokenizer is None:
            raise ValueError("Set `model.tokenizer` to use `stream=True`.")
        if stream:
            len_to_generate = range(max_gen_len)
            print(self.tokenizer.decode(idx[0].numpy().tolist()), end="", flush=True)
        else:
            len_to_generate = tqdm.tqdm(range(max_gen_len), desc="Generating text")
        cache = None
        for _ in len_to_generate:
            # if the sequence context is growing too long we must crop it at max_seq_len
            idx_cond = idx if idx.shape[1] <= self.config.max_position_embeddings \
                else idx[:, -self.config.max_position_embeddings:]

            logits, cache, _, _ = self(idx_cond, use_cache=False, training=False)
            logits = logits[:, -1, :]

            if temperature > 0.0:
                logits = logits / temperature

            if top_k > 0:
                top_k = min(top_k, self.config.vocab_size)
                indices_to_remove = logits < tf.math.top_k(logits, top_k)[0][..., -1, None]
                logits = tf.where(indices_to_remove, tf.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

            if 0.0 < top_p < 1.0:
                sorted_indices = tf.argsort(logits, direction='DESCENDING')
                sorted_logits = tf.gather(logits, sorted_indices, axis=-1, batch_dims=1)

                cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = tf.concat([tf.zeros_like(sorted_indices_to_remove[..., :1]),
                                                      sorted_indices_to_remove[..., :-1]], axis=-1)
                indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove, sorted_indices)
                logits = tf.where(indices_to_remove, tf.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

            probs = tf.nn.softmax(logits, axis=-1)
            idx_next = tfp.distributions.Categorical(probs=probs).sample()
            if stream:
                out = self.tokenizer.decode_piece(idx_next.numpy().tolist()[0])
                out = out.replace('▁', ' ')
                if match := re.match(r'<0x([0-9a-fA-F]+)>', out):
                    out = bytes.fromhex(match.group(1)).decode('utf-8')
                print(out, end='', flush=True)
            idx = tf.concat([idx, idx_next[:, tf.newaxis]], axis=1)

        if stream:
            print()

        return idx
