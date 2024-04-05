import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    The model configuration class.
    """
    vocab_size = 512
    hidden_size = 256
    intermediate_size = 1024
    num_hidden_layers = 1
    num_attention_heads = 4
    num_key_value_heads = 4
    is_moe = False
    sliding_window: Optional[int] = None
    num_local_experts = 1
    num_experts_per_tok = 1
    router_aux_loss_coef = 0.001
    hidden_act = "silu"
    max_position_embeddings = 128
    initializer_range = 0.02
    rms_norm_eps = 1e-06
    use_cache = True
    pad_token_id = 0
    rope_theta = 10000.0
    rope_scaling = None
    tie_word_embeddings = False
    attention_bias = False
    attention_dropout = 0.0
    gradient_checkpointing: Optional[str] = None  # 'mlp-only', 'attention-only', 'full', None
    grad_accumulation_steps = 1
    max_batch_size = 1
    epochs = 1
    is_encoder_decoder = False
    router_jitter_noise = 0.0

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
        if conf.num_local_experts == 1:
            conf.is_moe = False
        if conf.sliding_window is not None and conf.sliding_window < 1:
            conf.sliding_window = None
        if conf.is_moe and conf.num_experts_per_tok > conf.num_local_experts:
            raise ValueError("num_experts_per_tok must be less than or equal to num_local_experts.")
        return conf

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)


@dataclass
class LoRAConfig:
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_layers = ['q_proj', 'k_proj']  # 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'mlp'

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
