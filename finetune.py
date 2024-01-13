import os

import torch
from torch.utils.data import DataLoader

from finetune_datasets import CSVDataset, DollyDataset, HFnoRobotsDataset, AlpacaGpt4Dataset
from main import train
from model import ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file('weights/tokenizer.json')

    path = './finetuned-weights/'

    if os.path.exists(path + 'config.json'):
        params = ModelConfig.from_json(path + 'config.json')
        print("Loaded config from file.")
    else:
        raise ValueError("Config not found.")

    print("Loading datasets...")
    dataset1 = DollyDataset(params.max_position_embeddings, tokenizer)
    dataset2 = CSVDataset(path="data/finetune/DankDataset.csv",  # custom dataset
                          max_length=params.max_position_embeddings, tokenizer=tokenizer,
                          sample_frac=3.0)
    dataset3 = HFnoRobotsDataset(params.max_position_embeddings, tokenizer)
    dataset4 = AlpacaGpt4Dataset(params.max_position_embeddings, tokenizer)
    dataset = torch.utils.data.ConcatDataset([dataset2, dataset1, dataset3, dataset4])
    dataloader = DataLoader(dataset, batch_size=params.max_batch_size, shuffle=True, drop_last=True)
    print("Loaded datasets.")

    train(path,
          training_data=dataloader,
          config=params,
          disable_grads_for_embeddings=False,
          disable_scheduler=True,
          )
