import os
from datetime import date
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader

from finetune_datasets import HFnoRobotsDataset, CSVDatasetV2, WizardVicuna, JsonlConversations  # noqa
from main import train
from model import ModelConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    today = date.today()
    with open('data/finetune/sysprompt.txt', 'r') as f:
        sys_prompt = f.read()
    sys_prompt = sys_prompt.format(date=today.strftime("%B %d, %Y"))


    def collate_pad_batch_fn(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = tokenizer(batch, return_tensors='pt',
                            max_length=params.max_position_embeddings,
                            padding='max_length', truncation=True)
        return encoded['input_ids'], encoded['attention_mask']


    print("Loading datasets...")
    ds1 = CSVDatasetV2(path="data/finetune/DankDataset.csv",
                       sys_prompt=sys_prompt, tokenizer=tokenizer, max_seq_len=params.max_position_embeddings),
    ds2 = JsonlConversations(path="data/finetune/convos.jsonl", sys_prompt=sys_prompt, tokenizer=tokenizer,
                             max_seq_len=params.max_position_embeddings)
    dataset = torch.utils.data.ConcatDataset(
        [
            ds1, ds2,
            # WizardVicuna(sys_prompt,),
            HFnoRobotsDataset(sys_prompt, tokenizer, params.max_position_embeddings),
        ]
    )
    dataloader = DataLoader(dataset, batch_size=params.max_batch_size,
                            shuffle=True, collate_fn=collate_pad_batch_fn)

    validation_dataset = torch.utils.data.ConcatDataset([ds1, ds2])
    validation_dataloader = DataLoader(validation_dataset, batch_size=params.max_batch_size,
                                       shuffle=True, collate_fn=collate_pad_batch_fn, )
    print("Loaded datasets.")

    train(path,
          training_data=dataloader,
          validation_data=validation_dataloader,
          config=params,
          disable_grads_for_embeddings=False,
          disable_scheduler=True,
          learning_rate=1e-5,
          save_every=500,
          )
