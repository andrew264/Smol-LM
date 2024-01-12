import os

import torch
from torch.utils.data import DataLoader

from finetune_datasets import CSVDataset, DollyDataset
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

    dataset1 = DollyDataset(params.max_position_embeddings, tokenizer)
    dataset2 = CSVDataset(path="data/finetune/DankDataset.csv",
                          max_length=params.max_position_embeddings, tokenizer=tokenizer,
                          sample_frac=2.0)
    dataset = torch.utils.data.ConcatDataset([dataset2, dataset1])
    dataloader = DataLoader(dataset, batch_size=params.max_batch_size, shuffle=True, drop_last=True)

    train(path,
          training_data=dataloader,
          config=params,
          disable_grads_for_embeddings=False,
          disable_scheduler=True,
          )
