import json
from dataclasses import dataclass

from model.utils import find_multiple


@dataclass
class ModelConfig:
    """
    The model configuration class.
    """
    vocab_size = 32000
    hidden_size = 1024
    intermediate_size = 2560
    num_hidden_layers = 8
    num_attention_heads = 8
    num_key_value_heads = 4
    hidden_act = "silu"
    max_position_embeddings = 1024
    initializer_range = 0.02
    rms_norm_eps = 1e-6
    use_cache = True
    pad_token_id = None
    bos_token_id = 1
    eos_token_id = 2
    pretraining_tp = 1
    tie_word_embeddings = False
    rope_theta = 10000.0
    rope_scaling = None
    attention_bias = False
    attention_dropout = 0.0
    gradient_checkpointing = True

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """
        Loads a configuration from a json file.
        :param path: (str) The path to the json file.
        :return: (ModelConfig) The configuration class.
        """
        with open(path, "r") as f:
            return cls(**json.load(f))

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
