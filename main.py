import glob
import json
import os
import random

# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from keras.optimizers.schedules import CosineDecay

from model import SmolLM

# enable memory growth for GPU
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(f"Enabled TF_GPU_ALLOCATOR: {os.environ['TF_GPU_ALLOCATOR']}")
print(f"TF version: {tf.__version__}")
print(f"TF executing eagerly: {tf.executing_eagerly()}")

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
print(f"Global dtype policy: {tf.keras.mixed_precision.global_policy()}")

max_seq_len = 1024
vocab_size = 32000
batch_size = 4
dataset_path = './data/processed/*.bin'
logdir = r'./logs/'


def _generator(seq_len: int, path: str) -> tuple[tf.Tensor, tf.Tensor]:
    files = glob.glob(path, recursive=True)
    random.shuffle(files)
    for file_path in files:
        binary_data = tf.io.read_file(file_path)
        m = tf.io.decode_raw(binary_data, tf.uint16)
        num_batches = tf.shape(m)[0] // (seq_len + 1)
        m = m[:num_batches * (seq_len + 1)]  # Truncate to have an even number of batches
        m = tf.reshape(m, [num_batches, seq_len + 1])
        for i in range(num_batches):
            batch = m[i]
            yield batch[:-1], batch[1:]


def _get_total_steps(path: str) -> int:
    files = glob.glob(path, recursive=True)
    total_steps = 0
    for file_path in files:
        binary_data = tf.io.read_file(file_path)
        m = tf.io.decode_raw(binary_data, tf.uint16)
        num_batches: tf.Tensor = tf.shape(m)[0] // (max_seq_len + 1)
        total_steps += num_batches.numpy()
    return total_steps


if __name__ == '__main__':
    if os.path.exists(logdir):
        # remove old logs
        import shutil

        shutil.rmtree(logdir)
        print("Removed old logs.")

    dataset = tf.data.Dataset.from_generator(_generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
                                                 tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32)
                                             ),
                                             args=(max_seq_len, dataset_path))
    dataset = (dataset
               .shuffle(buffer_size=10000)
               .batch(batch_size=batch_size, drop_remainder=True)
               .prefetch(tf.data.experimental.AUTOTUNE)
               .repeat())
    total_steps = _get_total_steps(dataset_path) // batch_size
    print(f"Total Steps:- {total_steps}")
    print(f"Batch Size:- {batch_size}")

    # create graph for tensorboard
    # tf.summary.trace_on(graph=True, profiler=True)

    model = SmolLM(max_batch_size=batch_size, num_accumulation=8)
    warmup_steps = 5000
    initial_learning_rate = 6e-4
    final_learning_rate = 0.1 * initial_learning_rate
    beta_1 = 0.9
    beta_2 = 0.95
    epsilon = 1e-5
    weight_decay = 0.1
    clipnorm = 1.0
    lr_schedule = CosineDecay(initial_learning_rate=initial_learning_rate,
                              decay_steps=total_steps - warmup_steps,
                              warmup_target=final_learning_rate,
                              warmup_steps=warmup_steps)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule,
                                          beta_1=beta_1,
                                          beta_2=beta_2,
                                          epsilon=epsilon,
                                          weight_decay=weight_decay,
                                          clipvalue=clipnorm,)
    model.compile(optimizer=optimizer, jit_compile=True)
    model.build(input_shape=(batch_size, max_seq_len))
    model.summary()

    # force enable eager execution
    # tf.config.run_functions_eagerly(True)

    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')
    config = model.get_config()
    with open('./weights/config.json', 'w') as f:
        json.dump(config, f, indent=4)

    if os.path.exists('./weights/weights.hdf5'):
        model.load_weights('./weights/weights.hdf5')
        print("Weights Loaded from hdf5 file.")
    else:
        print("No weights found. Training from scratch.")

    checkpoint = tf.keras.callbacks.ModelCheckpoint('./weights/weights.hdf5',
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq=1500)
    print("Training Started.")
    model.fit(x=dataset, epochs=1, verbose=1, steps_per_epoch=total_steps,
              callbacks=[checkpoint])
    model.save_weights('./weights/weights.hdf5')
    print("Training Done.")
