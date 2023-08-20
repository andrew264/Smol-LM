import tensorflow as tf
from keras.layers import Embedding, Layer


class SharedOutput(Layer):
    """
    This is a custom output layer that shares its weights with an embedding layer.
    """

    def __init__(self, embedding_layer: Embedding, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer

    def update_weights(self, embedding: Embedding):
        """
        Updates the weights of the output layer to be the same as the embedding layer.
        :param embedding: (Embedding) The embedding layer.
        """
        self.embedding_layer = embedding

    def compute_output_shape(self, input_shape):
        return input_shape[0], tf.shape(self.embedding_layer.embeddings)[0]

    def call(self, inputs, *args, **kwargs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        output = tf.matmul(inputs, self.embedding_layer.embeddings, transpose_b=True, name="output_weights")
        return output
