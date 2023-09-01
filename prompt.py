import os
from typing import List

# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import ModelConfig, SmolLM, Tokenizer
from utils import multiline_input

import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
print(f"Global dtype policy: {tf.keras.mixed_precision.global_policy()}")

SYS_PROMPT = "You are a Helpful Assistant. " \
             "Follow the instructions below very carefully to complete the task. "

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

    if os.path.exists('./weights/checkpoint'):
        ckpt = tf.train.Checkpoint(model)
        ckpt.restore('./weights/weights.ckpt').expect_partial()
        print("Weights Loaded from ckpt file.")
    else:
        raise FileNotFoundError("Checkpoint file not found.")

    history: List[List[int]] = []

    while True:
        # if len of history is greater than 756, pop the first element
        history_length = sum([len(seq) for seq in history])
        if history_length > 900:
            history.pop(1)
        print('_' * 80)
        context = multiline_input()
        if not context or context == '':
            break
        if history:
            content = tokenizer.prepare_encode_instructions(context)
            tokenized = tokenizer.encode(content)
        else:
            content = tokenizer.prepare_encode_instructions(context, sys_prompt=SYS_PROMPT)
            tokenized = tokenizer.encode(content, bos=True, eos=False)
        history.append(tokenized)
        inp = [t for seq in history for t in seq]
        generated_tokens = model.generate([inp], max_gen_len=250,
                                          temperature=1.0, top_k=15, top_p=0.9, stream=True)
        history[-1].extend(generated_tokens)

        print()
