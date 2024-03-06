import os
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader

from finetune_datasets import CSVDataset, HFnoRobotsDataset, AlpacaGpt4Dataset, OpenInstruct, OrcaMath
from main import train
from model import ModelConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_pad_batch_fn(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(batch, return_tensors='pt',
                        max_length=2048, padding='longest',
                        pad_to_multiple_of=8)
    return encoded['input_ids'], encoded['attention_mask']


if __name__ == '__main__':
    from transformers import PreTrainedTokenizerFast

    tokenizer_path = 'weights/tokenizer.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token_id = 0

    path = './finetuned-weights/'

    if os.path.exists(path + 'config.json'):
        params = ModelConfig.from_json(path + 'config.json')
        print("Loaded config from file.")
    else:
        raise ValueError("Config not found.")

    print("Loading datasets...")
    dataset = torch.utils.data.ConcatDataset(
        [
            CSVDataset(path="data/finetune/DankDataset.csv", sample_frac=3.0),
            OpenInstruct(),
            OrcaMath(),
            AlpacaGpt4Dataset(),
            HFnoRobotsDataset(),
        ]
    )
    dataloader = DataLoader(dataset, batch_size=params.max_batch_size,
                            shuffle=True, collate_fn=collate_pad_batch_fn, num_workers=2, prefetch_factor=3)
    print("Loaded datasets.")

    train(path,
          training_data=dataloader,
          config=params,
          disable_grads_for_embeddings=False,
          disable_scheduler=True,
          learning_rate=1e-8,
          save_every=10000,
          )
