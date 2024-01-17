import json
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    The model configuration class.
    """
    vocab_size = -1
    hidden_size = 1024
    intermediate_size = 4096
    num_hidden_layers = 8
    num_attention_heads = 8
    num_key_value_heads = 8
    is_moe = True
    sliding_window = -1
    num_local_experts = 2
    num_experts_per_tok = 1
    router_aux_loss_coef = 0.001
    hidden_act = "gelu_new"
    max_position_embeddings = 1024
    initializer_range = 0.02
    rms_norm_eps = 1e-5
    use_cache = True
    pad_token_id = 0
    rope_theta = 10000.0
    rope_scaling = None
    attention_bias = False
    attention_dropout = 0.0
    gradient_checkpointing = True
    grad_accumulation_steps = 1
    max_batch_size = 1
    max_epochs = 1

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
        if conf.is_moe and conf.num_experts_per_tok > conf.num_local_experts:
            raise ValueError("num_experts_per_tok must be less than or equal to num_local_experts.")
        return conf

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
