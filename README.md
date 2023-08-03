# Smol LM

## A small language model for text generation based on LLAMA-2

This is a small language model for text generation based on LLAMA.<br>
I am just playing around with it and trying to understand how it works.<br>

### Changes made

- uses `mixed_bfloat16` policy in `tf.keras.mixed_precision.set_policy` to reduce memory usage
- uses custom `train_step` and `test_step` functions to also calculate the `perplexity` metric

### Usage

    - Run `python create_tokenizer.py` to create the tokenizer
    - Run `python prepare_datasets.py` to create the tfrecords
    - Run `python main.py` to train the model
    - Run `python generate_sequence.py` to generate text
