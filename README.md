# Smol LM

## A small language model for text generation in PyTorch

## Multimodality
### Experiments
[ImageExpts](https://github.com/andrew264/ImageExpts)</br>
[AudioExpts](https://github.com/andrew264/AudioExpts)


### Changes made

    - Flash Attention - 2 (from F.scaled_dot_product_attention)
    - Weight Decay Denylist for Norm, Bias and Embedding
    - PEFT with LoRA, LoRA+, DoRA
    - RAG for text generation
    - PyTorch Lightning Trainer
    - int8 quantization with bitsandbytes

### Supports loading the following pretrained weights:
    - Llama-2, Llama-3
    - Mistral

### Usage

    - Run `python main.py` to train the model
    - Run `python generate_sequence.py` to generate text
    - Run `python finetune.py` to fine-tune the model
    - Run `python prompt.py` to prompt the model after fine-tuning
