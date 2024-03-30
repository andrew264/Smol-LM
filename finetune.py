import datetime
import os
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader

from main import train, validate_model  # noqa
from model import ModelConfig, LoRAConfig
from utils import CSVDatasetV2, JsonlConversations, DiscordConversations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CROSS_ENTROPY_IGNORE_IDX = -100

if __name__ == '__main__':
    from tokenizers import Tokenizer

    tokenizer_path = 'weights/tokenizer.json'
    tokenizer = Tokenizer.from_file(path=tokenizer_path)

    path = './ft-weights/'

    if os.path.exists(path + 'config.json'):
        params = ModelConfig.from_json(path + 'config.json')
        print("Loaded config from file.")
    else:
        raise ValueError("Config not found.")

    lora_params = None
    if os.path.exists(path + 'lora.json'):
        lora_params = LoRAConfig.from_json(path + 'lora.json')
        print("Loaded LoRA config from file.")
    else:
        lora_params = LoRAConfig()
        print("Created new LoRA config.")
        lora_params.to_json(path + 'lora.json')

    dt = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    with open('data/finetune/sysprompt.txt', 'r') as f:
        sys_prompt = f.read()
    sys_prompt = sys_prompt.format(datetime=dt)


    def collate_pad_batch_fn(batch: List[Tuple[List[int], List[int]]]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        MAX_LEN = params.max_position_embeddings
        max_len = max([len(x[0]) for x in batch])
        input_ids = torch.stack([torch.tensor(x[0] + [0] * (max_len - len(x[0]))) for x in batch])
        labels = torch.stack([torch.tensor(x[1] + [CROSS_ENTROPY_IGNORE_IDX] * (max_len - len(x[1]))) for x in batch])
        attention_mask = (input_ids != 0).long()
        return input_ids[:, :MAX_LEN], labels[:, :MAX_LEN], attention_mask[:, :MAX_LEN]
        # return input_ids, labels, attention_mask


    print("Loading datasets...")
    ds1 = CSVDatasetV2(path="data/finetune/DankDataset.csv", tokenizer=tokenizer,
                       sys_prompt=sys_prompt)
    ds2 = JsonlConversations(path="data/finetune/convos.jsonl", tokenizer=tokenizer, sys_prompt=sys_prompt)
    ds3 = DiscordConversations(path="data/finetune/conversations", tokenizer=tokenizer, sys_prompt=sys_prompt)
    dataset = torch.utils.data.ConcatDataset([ds1, ds2, ds3])
    dataloader = DataLoader(dataset, batch_size=params.max_batch_size,
                            shuffle=True, collate_fn=collate_pad_batch_fn)
    print("Loaded datasets.")

    train(path,
          training_data=dataloader,
          validation_data=dataloader,
          config=params,
          lora_config=lora_params,
          disable_scheduler=True,
          learning_rate=1e-5,
          save_every=5000,
          )
    # validate_model(None, dataloader, True)
