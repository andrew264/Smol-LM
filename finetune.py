import os

import torch
from torch.utils.data import DataLoader

from finetune_datasets import CSVDataset, HFnoRobotsDataset, AlpacaGpt4Dataset, OpenInstruct, OrcaMath, SortedPadded
from main import train
from model import ModelConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    dataset1 = SortedPadded(OpenInstruct(params.max_position_embeddings, tokenizer), batch_size=params.max_batch_size)
    dataset2 = SortedPadded(CSVDataset(path="data/finetune/DankDataset.csv",  # custom dataset
                                       max_length=params.max_position_embeddings, tokenizer=tokenizer,
                                       sample_frac=3.0), batch_size=params.max_batch_size)
    dataset3 = SortedPadded(HFnoRobotsDataset(params.max_position_embeddings, tokenizer),
                            batch_size=params.max_batch_size)
    dataset4 = SortedPadded(AlpacaGpt4Dataset(params.max_position_embeddings, tokenizer),
                            batch_size=params.max_batch_size)
    dataset5 = SortedPadded(OrcaMath(params.max_position_embeddings, tokenizer), batch_size=params.max_batch_size)
    dataset = torch.utils.data.ConcatDataset([dataset2, dataset1, dataset3, dataset4, dataset5])
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True)
    print("Loaded datasets.")

    train(path,
          training_data=dataloader,
          config=params,
          disable_grads_for_embeddings=False,
          disable_scheduler=True,
          learning_rate=1e-7,
          save_every=10000,
          )
