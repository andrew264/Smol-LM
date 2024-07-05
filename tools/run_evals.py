import argparse
import json
import os

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedTokenizerFast

from utils import ModelGenerationHandler


def main(path: str, tasks: list[str]):
    print(f"Tasks: {', '.join(tasks)}")
    handler = ModelGenerationHandler(path, torch.device('cuda'), 1)
    handler.load_model(compiled=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, 'tokenizer.json'), eos_token="</s>",
                                        bos_token="<s>")
    model = HFLM(pretrained=handler.model, backend='causal', tokenizer=tokenizer, device=handler.device)
    results = simple_evaluate(model, tasks=tasks, batch_size=1)
    print(results['results'])
    with open(os.path.join(path, 'eval_results.json'), 'w') as f:
        json.dump(results['results'], f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run Evaluation on fine-tuned Models")
    parser.add_argument("--model", type=str, help="Path to the model", default="ft-weights/")
    parser.add_argument("--tasks", nargs='+', type=str, help="Tasks to evaluate on", default=["hellaswag"])
    args = parser.parse_args()

    main(args.model, args.tasks)
