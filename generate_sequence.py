import os.path

from utils import multiline_input

# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from model import SmolLM, Tokenizer, ModelConfig

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
print(f"Global dtype policy: {tf.keras.mixed_precision.global_policy()}")

if __name__ == '__main__':
    tokenizer = Tokenizer('./weights/tokenizer.model')
    if not os.path.exists('./weights/config.json'):
        print("No config file found. Exiting.")
        exit(1)
    config = ModelConfig.from_json('./weights/config.json')
    temperature = 1.0
    model = SmolLM(config=config)
    model.build(input_shape=(1, config.max_position_embeddings))
    print("Model Created.")
    model.tokenizer = tokenizer
    if os.path.exists('./weights/checkpoint'):
        ckpt = tf.train.Checkpoint(model)
        ckpt.restore('./weights/weights.ckpt').expect_partial()
        print("Weights Loaded from ckpt file.")
    else:
        print("No weights found. Exiting.")
        exit(1)
    model.summary()
    while True:
        context = multiline_input()
        if not context or context == '':
            break
        tokenized = tokenizer.encode([context], bos=True, eos=False)
        generated_seq = model.generate(tokenized, max_gen_len=150,
                                       temperature=temperature, top_k=8, stream=True)
        # generated_seq = tokenizer.decode(generated_seq[0].numpy().tolist())
        # print("Generated Sequence: ", generated_seq)
    print("kthxbye")
