import json
import os

import sentencepiece as spm
import tensorflow as tf
from tensorboard.plugins import projector

from model import SmolLM

if __name__ == '__main__':
    model_path = './weights/weights.hdf5'
    vocab = './weights/tokenizer.model'

    with open('./weights/config.json', 'r') as f:
        config = json.load(f)

    if not config:
        raise Exception("No config found")

    model = SmolLM(**config)
    model.build(input_shape=(1, config['max_seq_len']))

    if os.path.exists(model_path):
        model.load_weights(model_path)
        print("Loaded weights from hdf5")
    else:
        raise Exception("No weights found")

    sp = spm.SentencePieceProcessor()
    sp.load(vocab)

    embeddings = tf.Variable(model.transformer.get_embedding().embeddings)
    checkpoint = tf.train.Checkpoint(embedding=embeddings)
    checkpoint.save(os.path.join('./logs/', "embedding.ckpt"))
    print("Saved embeddings")

    with open(os.path.join('./logs/', 'metadata.tsv'), "w",) as f:
        for i in range(sp.vocab_size()):
            piece = sp.id_to_piece(i)
            f.write(piece + "\n")
    print("Saved metadata file")

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = './logs/metadata.tsv'
    projector.visualize_embeddings('./logs/', config)
