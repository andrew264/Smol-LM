import glob
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from main import train, NPDataset
from model import ModelConfig

dataset_paths = glob.glob('./data/processed-finetune/*.pkl', recursive=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PickleDataset(NPDataset):
    def __init__(self, path, block_size=1024):
        self.data = pickle.load(open(path, 'rb'))
        # pad 0 to the end of each sequence
        self.data = [x + [0] * ((block_size + 1) - len(x)) for x in self.data]


if __name__ == '__main__':

    path = './finetuned-weights/'

    if os.path.exists(path + 'config.json'):
        params = ModelConfig.from_json(path + 'config.json')
        print("Loaded config from file.")
    else:
        params = ModelConfig()
        params.vocab_size = 32000
        print("Created new config.")
        params.to_json(path + 'config.json')

    train_data = torch.utils.data.ConcatDataset([
        PickleDataset(path, params.max_position_embeddings) for path in dataset_paths
    ])
    dataloader = DataLoader(train_data, batch_size=params.max_batch_size, shuffle=True, drop_last=True)

    train(path,
          model_weights_path=path + 'model_ckpt.pt',
          training_data=dataloader,
          config=params,
          )
