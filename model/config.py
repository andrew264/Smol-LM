import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    The model configuration class.
    """
    vocab_size = 32000
    hidden_size = 1024
    intermediate_size = 3072
    num_hidden_layers = 8
    num_attention_heads = 8
    num_key_value_heads = 1
    is_moe = False
    sliding_window: Optional[int] = None
    num_experts = 1
    num_activated_experts = 1
    router_aux_loss_coef = 0.001
    hidden_act = "silu"
    max_position_embeddings = 1024
    initializer_range = 0.02
    rms_norm_eps = 1e-06
    use_cache = True
    pad_token_id = 0
    rope_theta = 10000.0
    rope_scaling = None
    tie_word_embeddings = False
    mlp_bias = False
    attention_qkv_bias = False
    attention_out_bias = False
    attention_dropout = 0.0
    partial_rotary_factor = 1.0
    gradient_checkpointing_percent: Optional[float] = .0
    grad_accumulation_steps = 1
    max_batch_size = 1
    epochs = 1
    is_encoder_decoder = False
    router_jitter_noise = 0.0
    block_types = ["attention", ]
    conv1d_width = 2
    lru_width = 256
    logits_soft_cap = None
    normalize_embedding = False

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
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_layers = ['qkv_proj']  # 'qkv_proj', 'o_proj', 'mlp', l_x, l_y, l_out

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
