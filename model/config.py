import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    The model configuration class.
    """
    # Basic Model Parameters
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_hidden_layers: int = 8

    # Huggingface compatibility
    is_encoder_decoder: bool = False
    use_cache: bool = True

    # Embeddings and Rotary Stuff
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    tie_word_embeddings: bool = False

    # Attention
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    qk_layernorm: bool = False
    swin_norm: bool = False

    # Feed Forward
    hidden_act: str = "silu"
    intermediate_size: int = 3072

    # Normalization
    rms_norm_eps: float = 1e-06

    # Dropout and Regularization
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    # Biases
    mlp_bias: bool = False
    attention_qkv_bias: bool = False
    attention_out_bias: bool = False

    # for training
    gradient_checkpointing_percent: Optional[float] = 0.0
    grad_accumulation_steps: int = 1
    max_batch_size: int = 1
    epochs: int = 1
    lr = 2e-5

    # Other Parameters
    pad_token_id: int = 0

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """
        Loads a configuration from a json file.
        :param path: (str) The path to the json file.
        :return: (ModelConfig) The configuration class.
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        conf = cls()
        for k, v in config_dict.items():
            setattr(conf, k, v)
        return conf

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def checkpointing_layers(self) -> list[int]:
        return [i for i in range(int(self.num_hidden_layers * self.gradient_checkpointing_percent))]


@dataclass
class LoRAConfig:
    type: str = 'lora'  # 'lora', 'dora'
    rank = 8
    alpha = 16
    dropout = 0.05
    layers = ['qkv_proj']  # 'qkv_proj', 'o_proj', 'mlp', lm_head, embedding

    @classmethod
    def from_json(cls, path: str) -> "LoRAConfig":
        """
        Loads a configuration from a json file.
        :param path: (str) The path to the json file.
        :return: (LoRAConfig) The configuration class.
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        conf = cls()
        for k, v in config_dict.items():
            setattr(conf, k, v)
        return conf

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def __repr__(self) -> str:
        return (f"LoRAConfig(rank={self.rank}, alpha={self.alpha}, "
                f"dropout={self.dropout}, layers={self.layers})")
