import os

import numpy as np

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

dataset_path = './data/processed/train.bin'
logdir = r'./logs/'


def _generator(seq_len: int, batch_size: int, s_step: int = 0) -> tuple[tf.Tensor, tf.Tensor]:
    steps = 0
    seq_len += 1
    binary_data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
    num_batches = len(binary_data) // seq_len
    binary_data = binary_data[:num_batches * seq_len]
    binary_data = binary_data.reshape(num_batches, seq_len)
    for i in range(num_batches):
        if s_step > 0:
            s_step -= 1
            steps += 1
            continue
        batch = tf.convert_to_tensor(binary_data[i])
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


def _get_embedding_and_head_weights(_config: ModelConfig):
    weights_path = './weights/llama2-7b'
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    embed_tokens = tf.Variable(tf.zeros(shape=(_config.vocab_size, _config.hidden_size), dtype=compute_dtype),
                               name='embed_tokens')
    ckpt = tf.train.Checkpoint(weights=embed_tokens)
    ckpt.restore(tf.train.latest_checkpoint(weights_path + '/embed_tokens'))
    lm_head = tf.Variable(tf.zeros(shape=(_config.vocab_size, _config.hidden_size), dtype=compute_dtype),
                          name='lm_head')
    ckpt = tf.train.Checkpoint(weights=lm_head)
    ckpt.restore(tf.train.latest_checkpoint(weights_path + '/lm_head'))
    bias = tf.Variable(tf.zeros(shape=(_config.vocab_size,), dtype=compute_dtype), name='bias')
    return embed_tokens, (tf.transpose(lm_head), bias)


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
    batch_size = config.batch_size
    use_llama2_weights = config.use_chopped_off_weights

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
                                             args=(max_seq_len, batch_size, start_step))
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

    model = SmolLM(config=config, num_accumulation=16)
    warmup_steps = 20000
    initial_learning_rate = 5e-4
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

    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')

    if os.path.exists('./weights/checkpoint'):
        model.load_weights('./weights/weights.ckpt')
        print("Weights Loaded from ckpt file.")
    else:
        print("No weights found. Training from scratch.")
        if use_llama2_weights:
            # the hacks begin here
            embed_weights, head_weights = _get_embedding_and_head_weights(config)
            model.transformer.set_embedding(embed_weights)
            model.transformer.set_lm_head(head_weights)
            # end of hacks
        start_step = 0

    if use_llama2_weights:
        # freeze those weights
        model.transformer.embed_tokens.trainable = False
        model.transformer.lm_head.trainable = False

    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint('./weights/weights.ckpt',
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq=2500)

    print("Training Started.")
    model.fit(x=dataset, steps_per_epoch=remaining_steps // batch_size,
              callbacks=[checkpoint], verbose=1, epochs=1)
    model.save_weights('./weights/weights.ckpt')
    print("Training Done.")
