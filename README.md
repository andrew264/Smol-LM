# Smol LM

## A small language model for text generation based on LLAMA-2 in PyTorch

This is a small language model for text generation based on LLAMA.<br>
use LLaMA-2's tokenizer. <br>

### Changes made
    - Added Gradient Checkpointing
    - Added Gradient Accumulation (with accelerate)
    - Added Mixture of Experts (MoE) from Mixtral

### Usage

    - Run `python main.py` to train the model
    - Run `python generate_sequence.py` to generate text
