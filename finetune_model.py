import json
import os

from keras.src.utils import pad_sequences

from utils import enable_memory_growth

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
dataset_path = './data/processed-finetune/finetuning-data.jsonl'
weights_path = './weights/fine-tuned/'


def _generator(seq_len: int, path: str) -> tuple[tf.Tensor, tf.Tensor]:
    seq_len += 1
    with open(path) as f:
        for line in f.readlines():
            data = json.loads(line)
            data = pad_sequences([data], maxlen=seq_len, padding='post', truncating='post')[0]
            data = tf.convert_to_tensor(data)
            yield data[:-1], data[1:]


def _get_total_steps(path: str) -> int:
    with open(path) as f:
        return len(f.readlines())


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
    use_llama2_weights = config.use_chopped_off_weights

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
    total_steps = _get_total_steps(dataset_path)
    print(f"Total Steps:- {total_steps // batch_size}")

    model = SmolLM(config)
    learning_rate = 1e-5
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,
                                          beta_1=0.9,
                                          beta_2=0.95,
                                          epsilon=1e-5,
                                          weight_decay=0.1,
                                          clipvalue=1.0, )
    model.compile(optimizer=optimizer, jit_compile=True)
    model.build(input_shape=(batch_size, max_seq_len))

    if os.path.exists(weights_path + 'checkpoint'):
        model.load_weights(weights_path + 'weights.ckpt')
        print("Loaded Fine-tuned weights.")
    elif os.path.exists('./weights/checkpoint'):
        model.load_weights('./weights/weights.ckpt')
        print("Loaded Pre-trained weights.")
    else:
        print("No checkpoint found. Exiting...")
        exit(1)

    if use_llama2_weights:
        # freeze those weights
        model.transformer.embed_tokens.trainable = False
        model.transformer.lm_head.trainable = False

    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_path + 'weights.ckpt',
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq=2500)

    print("Training Started.")

    model.fit(x=dataset, steps_per_epoch=total_steps // batch_size,
              callbacks=[checkpoint], verbose=1, epochs=2)
    model.save_weights(weights_path + 'weights.ckpt')
    print("Training Done.")
