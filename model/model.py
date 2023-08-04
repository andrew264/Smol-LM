from typing import Optional, List

import tensorflow as tf
import tensorflow_probability as tfp
import tqdm

from model.transformer import Transformer


class SmolLM(tf.keras.Model):
    """
    SmolLM model.

    Attributes:
        dim (int): The model dimensionality.
        n_layers (int): The number of layers.
        n_heads (int): The number of heads.
        vocab_size (int): The vocabulary size.
        max_batch_size (int): The maximum batch size.
        max_seq_len (int): The maximum sequence length.
        multiple_of (int): The multiple of the sequence length.
        ffn_dim_multiplier (float, optional): A multiplier for the feed-forward dimensionality.
                                                Default is None, which sets the multiplier to 4.0.
        norm_eps (float, optional): A small value used for numerical stability when normalizing.
                                    Default is 1e-5.
    """

    def __init__(self, dim: int = 768, n_layers: int = 8, n_heads: int = 8,
                 vocab_size: int = 32000, max_batch_size: int = 1, max_seq_len: int = 1024,
                 multiple_of: int = 256, ffn_dim_multiplier: Optional[float] = None,
                 norm_eps: float = 1e-05, **kwargs):
        super(SmolLM, self).__init__(**kwargs)
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps

        self.transformer = Transformer(dim=dim, n_layers=n_layers, n_heads=n_heads,
                                       vocab_size=vocab_size, max_seq_len=max_seq_len, max_batch_size=max_batch_size,
                                       multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier,
                                       norm_eps=norm_eps)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.perplexity_tracker = tf.keras.metrics.Mean(name="perplexity")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")

    def call(self, tokens: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Calls the model.

        :param tokens: (tf.Tensor) The input tokens tensor.
        :param kwargs: Additional keyword arguments.
        :return: The output logits tensor.
        """
        return self.transformer(tokens=tokens, **kwargs)

    def get_config(self):
        config = super(SmolLM, self).get_config()
        config.update({
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "multiple_of": self.multiple_of,
            "ffn_dim_multiplier": self.ffn_dim_multiplier,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        model = cls(**kwargs)
        model.load_weights(path)
        return model

    @staticmethod
    def get_padded_accuracy(labels, logits):
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

    @staticmethod
    def get_perplexity(cross_entropy):
        perplexity = tf.exp(cross_entropy)
        return perplexity

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
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self.get_loss(y, logits)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        y_pred = tf.nn.softmax(logits)
        self.loss_tracker.update_state(loss)
        perplexity = self.get_perplexity(loss)
        self.perplexity_tracker.update_state(perplexity)
        accuracy = self.get_padded_accuracy(y, y_pred)
        self.accuracy_tracker.update_state(accuracy)


        return {"loss": self.loss_tracker.result(),
                "perplexity": self.perplexity_tracker.result(),
                "accuracy": self.accuracy_tracker.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        logits = self(x, training=False)
        loss = self.get_loss(y, logits)

        self.loss_tracker.update_state(loss)
        perplexity = self.get_perplexity(loss)
        self.perplexity_tracker.update_state(perplexity)
        accuracy = self.get_padded_accuracy(y, logits)
        self.accuracy_tracker.update_state(accuracy)


        return {"loss": self.loss_tracker.result(),
                "perplexity": self.perplexity_tracker.result(),
                "accuracy": self.accuracy_tracker.result()}

    def generate(self, idx: List[List[int]], max_gen_len: int,
                 temperature: float = 0.6, top_k: Optional[int] = None) -> List[List[int]]:
        """
        Generates text from a prompt.

        Params:
        :param idx: A list of lists of integers. Each list of integers is a prompt.
        :param max_gen_len: The maximum length of the generated text.
        :param temperature: The temperature to use when sampling from the softmax distribution.
        :param top_k: The number of top tokens to consider when sampling from the softmax distribution.
        :return: A list of lists of integers. Each list of integers is a generated text.
        """
        for _ in tqdm.tqdm(range(max_gen_len)):
            # if the sequence context is growing too long we must crop it at max_seq_len
            idx_cond = idx if idx.shape[1] <= self.max_seq_len else idx[:, -self.max_seq_len:]

            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = tf.math.top_k(logits, k=1)
            else:
                logits = logits / temperature

                if top_k is not None:
                    values, indices = tf.math.top_k(logits, k=top_k)
                    min_value = tf.reduce_min(values, axis=-1, keepdims=True)
                    logits = tf.where(logits < min_value, tf.ones_like(logits) * -1e10, logits)

                probs = tf.nn.softmax(logits, axis=-1)
                idx_next = tfp.distributions.Categorical(probs=probs).sample()
            idx = tf.concat((idx, idx_next[:, tf.newaxis]), axis=1)

        return idx
