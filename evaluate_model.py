import os
from typing import Tuple

import numpy as np

from utils import get_total_steps, enable_memory_growth

# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from model import SmolLM, ModelConfig

# enable memory growth for GPU
enable_memory_growth()
print(f"TF version: {tf.__version__}")

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
print(f"Global dtype policy: {tf.keras.mixed_precision.global_policy()}")

dataset_path = './data/processed/val.bin'


def _iter_batch(seq_len: int) -> Tuple[tf.Tensor]:
    seq_len += 1
    binary_data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
    num_batches = len(binary_data) // seq_len
    binary_data = binary_data[:num_batches * seq_len]
    binary_data = binary_data.reshape(num_batches, seq_len)
    for i in range(num_batches):
        batch = tf.convert_to_tensor(binary_data[i])
        yield batch[:-1], batch[1:]


if __name__ == '__main__':

    if os.path.exists('./weights/config.json'):
        config = ModelConfig.from_json('./weights/config.json')
        print("Loaded config from file.")
    else:
        config = ModelConfig()
        print("Created new config.")
        config.to_json('./weights/config.json')

    max_seq_len = config.max_position_embeddings
    batch_size = config.batch_size

    dataset = tf.data.Dataset.from_generator(_iter_batch,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
                                                 tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32)
                                             ),
                                             args=(max_seq_len,))
    dataset = (dataset
               .shuffle(buffer_size=10000)
               .batch(batch_size=batch_size, drop_remainder=True)
               .prefetch(tf.data.experimental.AUTOTUNE)
               .repeat())
    total_steps = get_total_steps(dataset_path, max_seq_len)
    print(f"Total Steps:- {total_steps // batch_size}")
    print(f"Batch Size:- {batch_size}")

    model = SmolLM(config=config)
    model.compile(optimizer='adam', jit_compile=True)
    model.build(input_shape=(batch_size, max_seq_len))
    if os.path.exists('./weights/checkpoint'):
        model.load_weights('./weights/weights.ckpt')
    else:
        raise Exception("Weights not found.")

    model.summary()
    loss, perplexity, accuracy = model.evaluate(dataset, batch_size=batch_size, steps=total_steps // batch_size,
                                                verbose=1)
    print(f"Loss: {loss:.4f} | Perplexity: {perplexity:.4f} | Accuracy: {(accuracy * 100):.2f} %")
