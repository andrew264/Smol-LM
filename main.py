import glob
import os
import random

# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from keras.optimizers.schedules import CosineDecay

from model import SmolLM, ModelConfig

# enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# enable cuda_malloc_async allocator
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(f"Enabled TF_GPU_ALLOCATOR: {os.environ['TF_GPU_ALLOCATOR']}")
print(f"TF version: {tf.__version__}")

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
print(f"Global dtype policy: {tf.keras.mixed_precision.global_policy()}")

batch_size = 4
dataset_path = './data/processed/*.bin'
logdir = r'./logs/'


def _generator(seq_len: int, path: str, start_step: int = 0) -> tuple[tf.Tensor, tf.Tensor]:
    steps = 0
    seq_len += 1
    files = glob.glob(path, recursive=True)
    for file_path in files:
        binary_data = tf.io.read_file(file_path)
        m = tf.io.decode_raw(binary_data, tf.uint16)
        num_batches = tf.shape(m)[0] // seq_len
        if start_step > num_batches:
            start_step -= num_batches
            steps += num_batches
            print(f'Skipping file {file_path}...')
            continue
        print(f'Processing file {file_path}...')
        m = m[:num_batches * seq_len]  # Truncate to have an even number of batches
        m = tf.reshape(m, [num_batches, seq_len])
        for i in range(num_batches):
            if start_step > 0:
                start_step -= 1
                steps += 1
                continue
            batch = m[i]
            yield batch[:-1], batch[1:]
            if steps % 1000 == 0:
                with open('./weights/step.txt', 'w') as f:
                    if isinstance(steps, tf.Variable):
                        f.write(str(steps.value().numpy()))
                    elif isinstance(steps, tf.Tensor):
                        f.write(str(steps.numpy()))
                    else:
                        f.write(str(steps))


def _get_total_steps(path: str, seq_len: int) -> int:
    files = glob.glob(path, recursive=True)
    steps = 0
    for file_path in files:
        binary_data = tf.io.read_file(file_path)
        m = tf.io.decode_raw(binary_data, tf.uint16)
        num_batches: tf.Tensor = tf.shape(m)[0] // (seq_len + 1)
        steps += num_batches.numpy()
    return steps


if __name__ == '__main__':
    if os.path.exists(logdir):
        # remove old logs
        import shutil

        shutil.rmtree(logdir)
        print("Removed old logs.")

    if os.path.exists('./weights/config.json'):
        config = ModelConfig.from_json('./weights/config.json')
        print("Loaded config from file.")
    else:
        config = ModelConfig()
        print("Created new config.")
        config.to_json('./weights/config.json')
    max_seq_len = config.max_position_embeddings

    if os.path.exists('./weights/step.txt'):
        with open('./weights/step.txt', 'r') as f:
            try:
                start_step = int(f.read())
            except ValueError:
                start_step = 0
        print(f"Starting from step {start_step}")
    else:
        print("No step file found. Starting from step 0.")
        start_step = 0

    dataset = tf.data.Dataset.from_generator(_generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
                                                 tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32)
                                             ),
                                             args=(max_seq_len, dataset_path, start_step))
    dataset = (dataset
               # .shuffle(buffer_size=10000)
               .batch(batch_size=batch_size, drop_remainder=True)
               .prefetch(tf.data.experimental.AUTOTUNE)
               .repeat())
    total_steps = _get_total_steps(dataset_path, max_seq_len) // batch_size
    remaining_steps = total_steps - start_step
    print(f"Total Steps:- {total_steps}")
    print(f"Remaining Steps:- {remaining_steps}")
    print(f"Batch Size:- {batch_size}")

    # create graph for tensorboard
    # tf.summary.trace_on(graph=True, profiler=True)

    model = SmolLM(config=config, num_accumulation=8)
    warmup_steps = 5000
    initial_learning_rate = 6e-4
    final_learning_rate = 0.1 * initial_learning_rate
    beta_1 = 0.9
    beta_2 = 0.95
    epsilon = 1e-5
    weight_decay = 0.1
    clip_value = 1.0
    lr_schedule = CosineDecay(initial_learning_rate=initial_learning_rate,
                              decay_steps=total_steps - warmup_steps,
                              warmup_target=final_learning_rate,
                              warmup_steps=warmup_steps)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule,
                                          beta_1=beta_1,
                                          beta_2=beta_2,
                                          epsilon=epsilon,
                                          weight_decay=weight_decay,
                                          clipvalue=clip_value, )
    optimizer.iterations = tf.Variable(start_step // batch_size, dtype=tf.int64)
    model.compile(optimizer=optimizer, jit_compile=True)
    model.build(input_shape=(batch_size, max_seq_len))
    model.summary()

    # force enable eager execution
    # tf.config.run_functions_eagerly(True)

    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')

    if os.path.exists('./weights/weights.hdf5'):
        model.load_weights('./weights/weights.hdf5')
        print("Weights Loaded from hdf5 file.")
    else:
        print("No weights found. Training from scratch.")

    checkpoint = tf.keras.callbacks.ModelCheckpoint('./weights/weights.hdf5',
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq=2000)

    print("Training Started.")
    model.fit(x=dataset, steps_per_epoch=remaining_steps,
              callbacks=[checkpoint], verbose=1, epochs=1)
    model.save_weights('./weights/weights.hdf5')
    print("Training Done.")
