# Smol LM

## A small language model for text generation based on LLAMA-2 in Tensorflow 2.0

This is a small language model for text generation based on LLAMA.<br>
use LLaMA-2's tokenizer. <br>

### Changes made

- uses custom `train_step` and `test_step` functions to also calculate the `perplexity` metric
- uses Gradient Accumulation to increase the batch size
- All layers except Embeddings use `tf.bfloat16` to reduce memory usage (save weights as `ckpt` file).
- Optionally use Embedding weights from Bigger models (LLaMA2-7B) to initialize the Embedding and Head layers.

### Usage

    - Run `python main.py` to train the model
    - Run `python generate_sequence.py` to generate text
