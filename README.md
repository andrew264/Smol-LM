# Smol LM

## A small language model for text generation in PyTorch

### Changes made

    - Added Gradient Checkpointing
    - Added Gradient Accumulation (with accelerate)
    - Added Mixture of Experts (MoE) from Mixtral
    - Flash Attention - 2
    - Weight Decay Denylist for LayerNorm and Bias and Embedding

### Usage

    - Run `python main.py` to train the model
    - Run `python generate_sequence.py` to generate text
