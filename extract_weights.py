from safetensors import safe_open
import tensorflow as tf

if __name__ == '__main__':
    target_hidden_dim = 1024

    path = "./llama-2-13b/model-00001-of-00003.safetensors"  # put model file path here
    with safe_open(path, framework="tf", device="cpu") as f:
        tensor = f.get_slice("model.embed_tokens.weight")  # put layer name here
        vocab, hidden_dim = tensor.get_shape()

    n = hidden_dim // target_hidden_dim
    tensor = tensor[:, :n * target_hidden_dim]
    tensor = tf.cast(tensor, dtype=tf.float64)
    tensor = tf.reshape(tensor, shape=(vocab, target_hidden_dim, n))
    tensor = tf.reduce_mean(tensor, axis=-1)
    tf_data = tf.Variable(tensor, name="weights")
    checkpoint = tf.train.Checkpoint(weights=tf_data)
    checkpoint.save("./llama-2-13b/embed_tokens/embed_tokens.ckpt")  # put save path here
