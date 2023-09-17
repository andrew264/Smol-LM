import os
from typing import List

# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import ModelConfig, SmolLM, Tokenizer
from utils import multiline_input

import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
print(f"Global dtype policy: {tf.keras.mixed_precision.global_policy()}")

with open('./weights/sysprompt.txt') as f:
    SYS_PROMPT = f.read()

weights_path = './weights/fine-tuned/'

if __name__ == '__main__':
    if os.path.exists('./weights/config.json'):
        config = ModelConfig.from_json('./weights/config.json')
        print("Loaded config from file.")
    else:
        raise FileNotFoundError("Config file not found.")
    max_seq_len = config.max_position_embeddings

    model = SmolLM(config)
    tokenizer = Tokenizer('./weights/tokenizer.model')
    model.tokenizer = tokenizer

    if os.path.exists(weights_path + 'checkpoint'):
        ckpt = tf.train.Checkpoint(model)
        ckpt.restore(weights_path + 'weights.ckpt').expect_partial()
        print("Weights Loaded from ckpt file.")
    else:
        raise FileNotFoundError("Checkpoint file not found.")

    enable_history = input("Enable history? (y/n): ").lower() == 'y'
    history: List[List[int]] = []

    while True:
        # if len of history is greater than 80% of max_seq_len, pop the second element
        if enable_history:
            history_length = sum([len(seq) for seq in history])
            if history_length > int(0.8 * max_seq_len):
                if len(history) > 1:
                    history.pop(1)
                else:
                    history = []
        print('_' * 80)
        context = multiline_input()
        if not context or context == '':
            break
        if history:
            content = tokenizer.prepare_encode_instructions(context)
        else:
            content = tokenizer.prepare_encode_instructions(context, sys_prompt=SYS_PROMPT)
        tokenized = tokenizer.encode(content, bos=True)
        if enable_history:
            history.append(tokenized)
            inp = [t for seq in history for t in seq]
        else:
            inp = tokenized
        generated_tokens = model.generate([inp], max_gen_len=max_seq_len,
                                          temperature=1.0, top_k=12, repeat_penalty=1.3, stream=True)
        if enable_history:
            history[-1].extend(generated_tokens)

        print()
