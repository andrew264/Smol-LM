import os
import time
from typing import Optional, List, Tuple

import torch
from tokenizers import Tokenizer
from transformers import GenerationConfig, LogitsProcessorList, TopPLogitsWarper, StoppingCriteriaList

from model import ModelConfig, SmolLM, StaticCache, LoRAConfig, TemperatureRangeLogitsWarper
from .utils import get_state_dict_from_safetensors, StoppingCriteriaSub
from .lora_utils import inject_lora_adapter


class ModelGenerationHandler:
    def __init__(self, path: str, device: torch.device, num_beams: int):
        self.path = path
        self.num_beams = num_beams
        self.device = device
        self.config: Optional[ModelConfig] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[SmolLM] = None
        self.generation_config: Optional[GenerationConfig] = None
        self.cache: Optional[StaticCache] = None
        self.stopping_criteria = None
        self.processor: Optional[LogitsProcessorList] = None

    def load_model(self):
        self.config = ModelConfig.from_json(os.path.join(self.path, 'config.json'))
        self.config.max_batch_size = self.num_beams
        self.tokenizer = Tokenizer.from_file(os.path.join(self.path, 'tokenizer.json'))

        model_sd = get_state_dict_from_safetensors(os.path.join(self.path, 'model.safetensors'), self.device)
        self.model = SmolLM(self.config).to(device=self.device, dtype=torch.bfloat16)
        self.model.load_state_dict(model_sd)

        if os.path.exists(os.path.join(self.path, 'lora.json')):
            lora_params = LoRAConfig.from_json(os.path.join(self.path, 'lora.json'))

            adapter_sd = get_state_dict_from_safetensors(os.path.join(self.path, 'adapter.safetensors'), self.device)
            self.model = inject_lora_adapter(self.model, lora_params, adapter_sd)

        self.model.eval()
        torch.cuda.empty_cache()
        self.model.bos_token_id = self.tokenizer.token_to_id("<s>")

    def setup_generation(self, max_new_tokens: int = 512):
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=self.num_beams,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        self.cache = StaticCache(self.config, dtype=torch.bfloat16, device=self.device)

        stopping_tokens: List[torch.Tensor] = [torch.tensor([i], device=self.device) for i in range(3)]
        stopping_tokens.append(torch.tensor([523, 28766], device=self.device))
        stopping_tokens.append(torch.tensor([28789, 28766], device=self.device))
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])

    def setup_processor(self, top_p: float = 0.99, temperature: float = 1.7):
        self.processor = LogitsProcessorList([
            TemperatureRangeLogitsWarper(temperature, 0.8, 12),
            TopPLogitsWarper(top_p=top_p)
        ])

    def compile_model(self):
        print('Compiling...')
        start = time.time()
        self.model._forward = torch.compile(self.model._forward, mode="reduce-overhead")

        # Dummy run for compilation
        inp = "Love is a beautiful and"
        encoded = self.tokenizer.encode(inp, add_special_tokens=False)
        tokens = torch.tensor([encoded.ids]).to(self.device)
        attention_mask = torch.tensor([encoded.attention_mask]).to(self.device)
        self.model.generate(input_ids=tokens,
                            attention_mask=attention_mask,
                            past_key_values=self.cache,
                            generation_config=self.generation_config)

        print(f'Compiled in {time.time() - start:.3f}s')

    def generate(self, prompt: str) -> Tuple[str, int, int, float]:
        """
        Generate a completion for a given prompt
        :param prompt: The prompt to generate a completion for
        :return: The generated completion, number of tokens generated, total tokens including prompt, generation time

        """
        self.cache.reset()
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = torch.tensor([encoded.ids]).to(self.device)
        attention_mask = torch.tensor([encoded.attention_mask]).to(self.device)

        start = time.time()
        out = self.model.generate(input_ids=tokens,
                                  attention_mask=attention_mask,
                                  logits_processor=self.processor,
                                  past_key_values=self.cache,
                                  generation_config=self.generation_config,
                                  stopping_criteria=self.stopping_criteria)[0].tolist()

        total_tokens = len(out)
        out_tokens = out[len(encoded.ids):]
        decoded = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
        generation_time = time.time() - start

        return decoded, len(out_tokens), total_tokens, generation_time
