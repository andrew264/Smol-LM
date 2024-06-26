import gc
import os
import time
from typing import Optional, List, Tuple

import torch
from tokenizers import Tokenizer
from transformers import GenerationConfig, LogitsProcessorList, TopPLogitsWarper, StoppingCriteriaList

from model import ModelConfig, SmolLM, StaticCache, LoRAConfig, TemperatureRangeLogitsWarper
from .lora_utils import inject_lora_adapter
from .utils import get_state_dict_from_safetensors, StoppingCriteriaSub


class ModelGenerationHandler:
    def __init__(self, path: str, device: torch.device, num_beams: int):
        self.path = path
        self.num_beams = num_beams
        self.device = device
        self.config: Optional[ModelConfig] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[SmolLM] = None
        self.cache: Optional[StaticCache] = None
        self.stopping_criteria = self._get_stop_criteria()
        self.processor: Optional[LogitsProcessorList] = None
        self.set_processor()

    def load_model(self, compiled: bool = False):
        self.config = ModelConfig.from_json(os.path.join(self.path, 'config.json'))
        self.config.max_batch_size = self.num_beams
        self.tokenizer = Tokenizer.from_file(os.path.join(self.path, 'tokenizer.json'))

        model_sd = get_state_dict_from_safetensors(os.path.join(self.path, 'model.safetensors'), self.device)
        self.model = SmolLM(self.config).to(device=self.device, dtype=torch.bfloat16)
        self.model.load_state_dict(model_sd)
        del model_sd

        if os.path.exists(os.path.join(self.path, 'lora.json')):
            lora_params = LoRAConfig.from_json(os.path.join(self.path, 'lora.json'))

            adapter_sd = get_state_dict_from_safetensors(os.path.join(self.path, 'adapter.safetensors'), self.device)
            self.model = inject_lora_adapter(self.model, lora_params, adapter_sd)
            del adapter_sd

        self.model.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.model.eval()
        gc.collect()

        self.cache = StaticCache(self.config, compiled_mode=compiled, device=self.device)
        if compiled:
            self._compile_model()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=self.device)

    def get_gen_config(self, max_new_tokens: int = 512):
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=self.num_beams,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

    def _get_stop_criteria(self, ):
        stopping_tokens: List[torch.Tensor] = [torch.tensor([i], device=self.device) for i in range(3)]
        stopping_tokens.append(torch.tensor([523, 28766], device=self.device))
        stopping_tokens.append(torch.tensor([28789, 28766], device=self.device))
        return StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])

    def set_processor(self, top_p: float = 0.99, temperature: float = 1.7):
        self.processor = LogitsProcessorList([
            TemperatureRangeLogitsWarper(temperature, 0.8, 12),
            TopPLogitsWarper(top_p=top_p)
        ])

    def _compile_model(self):
        print('Compiling...')
        start = time.time()
        self.model.forward = torch.compile(self.model.forward, fullgraph=True, mode="reduce-overhead")

        # Dummy run for compilation
        inp = "Love is a beautiful and"
        for _ in range(2):  # Run twice cuz idl why; but this works? somehow?
            self.generate(inp, max_new_tokens=10)

        print(f'Compiled in {time.time() - start:.3f}s')

    def generate(self, prompt: str, max_new_tokens: int = 512) -> Tuple[str, int, int, float]:
        """
        Generate a completion for a given prompt
        :param prompt: The prompt to generate a completion for
        :param max_new_tokens: The maximum number of tokens to generate
        :return: The generated completion, number of tokens generated, total tokens including prompt, generation time

        """
        self.cache.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = torch.tensor([encoded.ids]).to(self.device)
        attention_mask = torch.tensor([encoded.attention_mask]).to(self.device)

        start = time.time()
        out = self.model.generate(input_ids=tokens,
                                  attention_mask=attention_mask,
                                  logits_processor=self.processor,
                                  past_key_values=self.cache,
                                  generation_config=self.get_gen_config(max_new_tokens=max_new_tokens),
                                  stopping_criteria=self.stopping_criteria)[0].tolist()

        total_tokens = len(out)
        out_tokens = out[len(encoded.ids):]
        decoded = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
        generation_time = time.time() - start

        return decoded, len(out_tokens), total_tokens, generation_time
