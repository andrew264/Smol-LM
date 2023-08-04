import json
import os.path

import tensorflow as tf

from model import SmolLM, Tokenizer

# disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    model_path = './weights/weights.hdf5'
    vocab = './weights/tokenizer.model'
    seq_len = 1024
    temperature = 1.0
    params = json.load(open('./weights/config.json', 'r'))
    model = SmolLM(**params)
    model.build(input_shape=(1, seq_len))
    print("Model Created.")
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print("Weights Loaded from hdf5 file.")
    else:
        print("No weights found. Exiting.")
    model.summary()
    tokenizer = Tokenizer(vocab)
    while True:
        context = input("Enter context: ") or None
        if not context:
            break
        tokenized = tokenizer.encode([context], bos=True, eos=False)
        generated_seq = model.generate(tf.constant(tokenized, dtype=tf.int32), max_gen_len=150,
                                       temperature=temperature, top_k=8)
        generated_seq = tokenizer.decode(generated_seq[0].numpy().tolist())
        print("Generated Sequence: ", generated_seq)
    print("kthxbye")
