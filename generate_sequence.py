import os.path

import tensorflow as tf

from model import SmolLM, Tokenizer, ModelConfig

# disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def multiline_input(prompt: str = '>> ') -> str:
    lines = []
    while True:
        line = input(prompt).strip()
        if line:
            lines.append(line)
        else:
            break
    return '\n'.join(lines)


if __name__ == '__main__':
    model_path = './weights/weights.hdf5'
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
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print("Weights Loaded from hdf5 file.")
    else:
        print("No weights found. Exiting.")
        exit(1)
    model.summary()
    while True:
        context = multiline_input()
        if not context or context == '':
            break
        print('_' * 80)
        tokenized = tokenizer.encode([context], bos=True, eos=False)
        generated_seq = model.generate(tf.constant(tokenized, dtype=tf.int32), max_gen_len=150,
                                       temperature=temperature, top_k=8, stream=True)
        # generated_seq = tokenizer.decode(generated_seq[0].numpy().tolist())
        # print("Generated Sequence: ", generated_seq)
    print("kthxbye")
