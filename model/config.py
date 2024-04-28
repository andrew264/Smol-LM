import json
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """
    The model configuration class.
    """
    # Basic Model Parameters
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 8

    # Huggingface compatibility
    is_encoder_decoder: bool = False
    use_cache: bool = True

    # Embeddings and Rotary Stuff
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    normalize_embedding: bool = False
    partial_rotary_factor: float = 1.0
    tie_word_embeddings: bool = False

    # Attention
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    sliding_window: Optional[int] = None

    # Feed Forward / Mixture of Experts
    hidden_act: str = "silu"
    is_moe: bool = False
    num_experts: int = 1
    num_activated_experts: int = 1
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    # Normalization
    rms_norm_eps: float = 1e-06
    use_gemma_rms_norm: bool = False

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

    # Other Parameters
    pad_token_id: int = 0
    block_types: List[str] = field(default_factory=lambda: ["attention"])
    logits_soft_cap: Optional[float] = None

    # Real-Gated Linear Recurrent Unit Parameters
    conv1d_width: int = 2
    lru_width: int = 256

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
        if conf.num_experts == 1:
            conf.is_moe = False
        if conf.sliding_window is not None and conf.sliding_window < 1:
            conf.sliding_window = None
        if conf.is_moe and conf.num_activated_experts > conf.num_experts:
            raise ValueError("num_activated_experts must be less than or equal to num_experts.")
        return conf

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def layers_block_types(self):
        return [self.block_types[i % len(self.block_types)] for i in range(self.num_hidden_layers)]

    @property
    def checkpointing_layers(self) -> list[int]:
        return [i for i in range(int(self.num_hidden_layers * self.gradient_checkpointing_percent))]


@dataclass
class LoRAConfig:
    rank = 8
    alpha = 16
    dropout = 0.05
    layers = ['qkv_proj']  # 'qkv_proj', 'o_proj', 'mlp', l_x, l_y, l_out, lm_head, embedding

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
