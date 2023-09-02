# Smol LM

## A small language model for text generation based on LLAMA-2 in Tensorflow 2.0

This is a small language model for text generation based on LLAMA.<br>
use LLaMA-2's tokenizer. <br>

### Changes made

- uses custom `train_step` and `test_step` functions to also calculate the `perplexity` metric
- uses Gradient Accumulation to increase the batch size
- output layer reuses weights from input embedding layer (reducing the memory footprint).
- All layers uses `tf.bfloat16` to reduce memory usage (save weights as `ckpt` file).

### Usage

    - Run `python prepare_datasets.py` to create the datasets
    - Run `python main.py` to train the model
    - Run `python generate_sequence.py` to generate text
