import os

import sentencepiece as spm

# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorboard.plugins import projector

from model import SmolLM, ModelConfig

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

if __name__ == '__main__':
    vocab = './weights/tokenizer.model'

    if os.path.exists('./weights/config.json'):
        config = ModelConfig.from_json('./weights/config.json')
        print("Loaded config from file.")
    else:
        raise Exception("No config found")

    model = SmolLM(config=config)
    model.build(input_shape=(1, config.max_position_embeddings))

    if os.path.exists('./weights/checkpoint'):
        ckpt = tf.train.Checkpoint(model)
        ckpt.restore('./weights/weights.ckpt').expect_partial()
        print("Loaded weights from ckpt file.")
    else:
        raise Exception("No weights found")

    sp = spm.SentencePieceProcessor()
    sp.load(vocab)

    embeddings = tf.Variable(tf.cast(model.transformer.get_embedding().embeddings, dtype=tf.float32))
    checkpoint = tf.train.Checkpoint(embedding=embeddings)
    checkpoint.save(os.path.join('./logs/', "embedding.ckpt"))
    print("Saved embeddings")

    with open(os.path.join('./logs/', 'metadata.tsv'), "w", ) as f:
        for i in range(sp.vocab_size()):
            piece = sp.id_to_piece(i)
            f.write(f'{i}-{piece}\n')
    print("Saved metadata file")

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings('./logs/', config)
