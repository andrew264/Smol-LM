import gc
import os
import time
from typing import Optional, List, Tuple, Union

import torch
from tokenizers import Tokenizer
from transformers import GenerationConfig, LogitsProcessorList, TopPLogitsWarper, StoppingCriteriaList, \
    InfNanRemoveLogitsProcessor

from model.config import ModelConfig, LoRAConfig
from model.peft.utilities import inject_lora_adapter
from model.transformer import SmolLM
from utils.utils import get_state_dict_from_safetensors
from .cache import StaticCache
from .sampling import TemperatureRangeLogitsWarper, StoppingCriteriaSub


class ModelGenerationHandler:
    def __init__(self, path: str, device: Union[str | torch.device], num_beams: int):
        self.path = path
        self.num_beams = num_beams
        self.device = torch.device(device) if isinstance(device, str) else device
        self.config: Optional[ModelConfig] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[SmolLM] = None
        self.cache: Optional[StaticCache] = None
        self.stopping_criteria = self._get_stop_criteria()
        self.processor: Optional[LogitsProcessorList] = None
        self.set_processor()

    def load_model(self, compiled: bool = False, merge_lora: bool = True,
                   load_in_8bit: bool = False, load_in_4bit: bool = False):
        if load_in_4bit and load_in_8bit:
            print("Can't load in both 4bit and 8bit. Loading model in 4bits")
            load_in_8bit = False
        self.config = ModelConfig.from_json(os.path.join(self.path, 'config.json'))
        self.config.max_batch_size = self.num_beams
        self.tokenizer = Tokenizer.from_file(os.path.join(self.path, 'tokenizer.json'))

        if os.path.exists(os.path.join(self.path, 'model.safetensors')):
            model_sd = get_state_dict_from_safetensors(os.path.join(self.path, 'model.safetensors'), torch.device('cpu'))
        else: raise FileNotFoundError("Model file not found.")

        model = SmolLM(self.config).bfloat16()
        model.load_state_dict(model_sd, assign=True)
        del model_sd

        if os.path.exists(os.path.join(self.path, 'lora.json')):
            lora_params = LoRAConfig.from_json(os.path.join(self.path, 'lora.json'))
            adapter_sd = get_state_dict_from_safetensors(os.path.join(self.path, 'adapter.safetensors'), torch.device('cpu'))
            inject_lora_adapter(model, lora_params, adapter_sd, merge_lora=merge_lora)
            del adapter_sd

        model.bos_token_id = self.tokenizer.token_to_id("<s>")
        model.eval()
        model.to(device=self.device)
        gc.collect()

        if load_in_8bit:
            model.to_8bit()
        elif load_in_4bit:
            model.to_4bit()

        self.model = model

        self.cache = StaticCache(self.config, compiled_mode=compiled, device=self.device)
        if compiled: self._compile_model()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device=self.device)

        self.model.generation_config = self.get_gen_config(None)

    def get_gen_config(self, max_new_tokens: Optional[int] = 512):
        return GenerationConfig(max_new_tokens=max_new_tokens, do_sample=True, num_beams=self.num_beams, use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2)

    def _get_stop_criteria(self, ):
        stopping_tokens: List[torch.Tensor] = [torch.tensor([i], device=self.device) for i in range(3)]
        stopping_tokens.append(torch.tensor([523, 28766], device=self.device))
        stopping_tokens.append(torch.tensor([28789, 28766], device=self.device))
        return StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])

    def set_processor(self, top_p: float = 0.95, temperature: float = 1.7):
        self.processor = LogitsProcessorList([TemperatureRangeLogitsWarper(temperature, 0.9, 24), TopPLogitsWarper(top_p=top_p), InfNanRemoveLogitsProcessor()])

    def _compile_model(self):
        print('Compiling...')
        start = time.time()
        self.model.forward = torch.compile(self.model.forward, fullgraph=True, mode="reduce-overhead")

        # Dummy run for compilation
        inp = "Love is a beautiful and"
        for _ in range(2): self.generate(inp, max_new_tokens=10)  # Run twice cuz idl why; but this works? somehow?

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
        if self.device.type == 'cuda':
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
