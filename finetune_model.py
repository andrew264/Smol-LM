import glob
import os

from utils import get_total_steps, enable_memory_growth

# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# enable gpu_private thread mode
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf

from model import SmolLM, ModelConfig

# enable memory growth for GPU
enable_memory_growth()
print(f"TF version: {tf.__version__}")

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
print(f"Global dtype policy: {tf.keras.mixed_precision.global_policy()}")
batch_size = 4
dataset_path = './data/processed-finetune/*.bin'


def _generator(seq_len: int, path: str) -> tuple[tf.Tensor, tf.Tensor]:
    seq_len += 1
    files = glob.glob(path, recursive=True)
    for file_path in files:
        binary_data = tf.io.read_file(file_path)
        m = tf.io.decode_raw(binary_data, tf.uint16)
        num_batches = tf.shape(m)[0] // seq_len
        print(f'Processing file {file_path}...')
        m = m[:num_batches * seq_len]  # Truncate to have an even number of batches
        m = tf.reshape(m, [num_batches, seq_len])
        for i in range(num_batches):
            batch = m[i]
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
               .repeat()
               )
    total_steps = get_total_steps(dataset_path, max_seq_len)
    print(f"Total Steps:- {total_steps // batch_size}")

    model = SmolLM(config)
    learning_rate = 5e-5
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,
                                          beta_1=0.9,
                                          beta_2=0.95,
                                          epsilon=1e-5,
                                          weight_decay=0.1,
                                          clipvalue=1.0, )
    model.compile(optimizer=optimizer, jit_compile=True)
    model.build(input_shape=(batch_size, max_seq_len))
    model.summary()

    if os.path.exists('./weights/checkpoint'):
        model.load_weights('./weights/weights.ckpt')
        print("Weights Loaded from ckpt file.")
    else:
        print("No checkpoint found. Exiting...")
        exit(1)

    checkpoint = tf.keras.callbacks.ModelCheckpoint('./weights/weights.ckpt',
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq=2500)

    print("Training Started.")

    model.fit(x=dataset, steps_per_epoch=total_steps // batch_size,
              callbacks=[checkpoint], verbose=1, epochs=2)
    model.save_weights('./weights/weights.ckpt')
    print("Training Done.")
