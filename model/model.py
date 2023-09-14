import re
from typing import List, Optional

import tensorflow as tf
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
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                         reduction=tf.keras.losses.Reduction.NONE
                                                                         )
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

    @tf.function
    def update_metrics(self, loss, logits, labels):
        y_pred = tf.nn.softmax(logits)
        self.loss_tracker.update_state(loss)
        self.perplexity_tracker.update_state(self.get_perplexity(loss))
        accuracy = self.get_padded_accuracy(labels, y_pred)
        self.accuracy_tracker.update_state(accuracy)

    @tf.function(jit_compile=True)
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self.get_loss(y, logits)

        gradients = tape.gradient(loss, self.trainable_variables)

        if self.num_accumulation == 1:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.update_metrics(loss, logits, y)
        else:
            self._gradient_accumulator(gradients)
            if self._gradient_accumulator.step == self.num_accumulation:
                averaged_gradients = [gradient / tf.cast(self.num_accumulation, dtype=gradient.dtype) for gradient in
                                      self._gradient_accumulator.gradients]
                self.optimizer.apply_gradients(zip(averaged_gradients, self.trainable_variables))
                self._gradient_accumulator.reset()
                self.update_metrics(loss, logits, y)

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
                 ) -> List[List[int]] or List[int]:
        """
        Generates text from a prompt.

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

        len_to_generate = range(max_gen_len) if stream else tqdm.trange(max_gen_len, desc="Generating text")

        generated_tokens = []
        for _ in len_to_generate:
            # if the sequence context is growing too long we must crop it at max_seq_len
            idx_cond = idx if len(idx[0]) <= self.config.max_position_embeddings else idx[:,
                                                                                      -self.config.max_position_embeddings:]
            logits = self(idx_cond, training=False)
            logits = tf.cast(logits[:, -1, :], dtype=tf.float32)

            if temperature > 0.0:
                logits = logits / temperature

            if top_k > 0:
                top_k = min(top_k, self.config.vocab_size)
                min_values = tf.nn.top_k(logits, k=top_k, sorted=True)[0][:, -1]
                logits = tf.where(logits < min_values, tf.ones_like(logits) * -1e10, logits)

            if 0.0 < top_p < 1.0:
                sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
                cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
                less_than_top_p = tf.cast(cumulative_probs <= top_p, tf.int32)
                indices = tf.argmax(less_than_top_p, axis=-1, output_type=tf.int32) - 1
                min_values = tf.gather_nd(sorted_logits, tf.stack([tf.range(tf.shape(indices)[0]), indices], axis=-1))
                logits = tf.where(logits < min_values, tf.fill(logits.shape, -1e10), logits)

            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            generated_tokens.append(samples[0, 0].numpy())
            if samples[0, 0] == self.tokenizer.bos_id or samples[0, 0] == self.tokenizer.eos_id:
                break
            if stream:
                out = self.tokenizer.decode_piece(samples[0, 0].numpy().tolist())
                out = out.replace('▁', ' ')
                if match := re.match(r'<0x([0-9a-fA-F]+)>', out):
                    out = bytes.fromhex(match.group(1)).decode('utf-8')
                print(out, end='', flush=True)
            idx = tf.concat([idx, samples], axis=-1)

        if stream:
            return generated_tokens

        return idx
