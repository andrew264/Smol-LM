import json
from dataclasses import dataclass
from typing import Optional, Self


@dataclass
class ModelConfig:
    """
    The model configuration class.
    """
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 768
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    max_position_embeddings: int = 1024
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    num_key_value_heads: int = 2
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-05
    vocab_size: int = 32000

    @classmethod
    def from_json(cls, path: str) -> Self:
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
