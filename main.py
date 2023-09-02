import glob
import os

from utils import get_total_steps, enable_memory_growth

# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# enable gpu_private thread mode
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from keras.optimizers.schedules import CosineDecay

from model import SmolLM, ModelConfig

# enable memory growth for GPU
enable_memory_growth()
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
            steps += 1
            yield batch[:-1], batch[1:]
            if steps % (2500 * batch_size) == 0:
                with open('./weights/step.txt', 'w') as f:
                    if isinstance(steps, tf.Variable):
                        f.write(str(steps.value().numpy()))
                    elif isinstance(steps, tf.Tensor):
                        f.write(str(steps.numpy()))
                    else:
                        f.write(str(steps))


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
    total_steps = get_total_steps(dataset_path, max_seq_len)
    remaining_steps = total_steps - start_step
    print(f"Total Steps:- {total_steps // batch_size}")
    print(f"Remaining Steps:- {remaining_steps // batch_size}")
    print(f"Batch Size:- {batch_size}")

    # create graph for tensorboard
    # tf.summary.trace_on(graph=True, profiler=True)

    # force enable eager execution
    # tf.config.run_functions_eagerly(True)

    model = SmolLM(config=config, num_accumulation=1)
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

    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')

    if os.path.exists('./weights/checkpoint'):
        model.load_weights('./weights/weights.ckpt')
        print("Weights Loaded from ckpt file.")
    else:
        print("No weights found. Training from scratch.")
        start_step = 0

    checkpoint = tf.keras.callbacks.ModelCheckpoint('./weights/weights.ckpt',
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq=2500)

    print("Training Started.")
    model.fit(x=dataset, steps_per_epoch=remaining_steps // batch_size,
              callbacks=[checkpoint], verbose=1, epochs=1)
    model.save_weights('./weights/weights.ckpt')
    print("Training Done.")
