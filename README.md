# Smol LM

## A small language model for text generation in PyTorch

### Changes made

    - Added Gradient Checkpointing
    - Added Gradient Accumulation (with accelerate)
    - Added Mixture of Experts (MoE) from Mixtral
    - Flash Attention - 2 (from F.scaled_dot_product_attention)
    - With Recurrent Blocks (from recurrent gemma)
    - Weight Decay Denylist for Norm, Bias and Embedding
    - Added LoRA fine-tuning
    - Added RAG for text generation

### Usage

    - Run `python main.py` to train the model
    - Run `python generate_sequence.py` to generate text
    - Run `python finetune.py` to fine-tune the model
    - Run `python prompt.py` to prompt the model after fine-tuning
