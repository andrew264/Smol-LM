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
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 2560
    multiple_of: int = 256
    max_position_embeddings: int = 1024
    num_attention_heads: int = 8
    num_hidden_layers: int = 8
    num_key_value_heads: int = 8
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-05
    vocab_size: int = 32000
    rope_scaling: Optional[dict] = None  # {"type": "dynamic", factor: 1.0, }
    tie_word_embeddings: bool = False
    use_chopped_off_weights: bool = False  # Use Embedding and LM Head weights from the original LLAMA2-7B model.
    batch_size: int = 4

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
