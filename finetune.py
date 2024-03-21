import os
from datetime import date
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader

from main import train
from model import ModelConfig, LoRAConfig
from utils import HFnoRobotsDataset, CSVDatasetV2, WizardVicuna, JsonlConversations, SmallOrca  # noqa

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    from transformers import PreTrainedTokenizerFast

    tokenizer_path = 'weights/tokenizer.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token_id = 0

    path = './ft-weights/'

    if os.path.exists(path + 'config.json'):
        params = ModelConfig.from_json(path + 'config.json')
        print("Loaded config from file.")
    else:
        raise ValueError("Config not found.")

    if os.path.exists(path + 'lora.json'):
        lora_params = LoRAConfig.from_json(path + 'lora.json')
        print("Loaded LoRA config from file.")
    else:
        lora_params = LoRAConfig()
        print("Created new LoRA config.")
        lora_params.to_json(path + 'lora.json')

    today = date.today()
    with open('data/finetune/sysprompt.txt', 'r') as f:
        sys_prompt = f.read()
    sys_prompt = sys_prompt.format(date=today.strftime("%B %d, %Y"))


    def collate_pad_batch_fn(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = tokenizer(batch, return_tensors='pt',
                            max_length=params.max_position_embeddings,
                            padding='longest',
                            pad_to_multiple_of=8,
                            truncation=True)
        return encoded['input_ids'], encoded['attention_mask']


    print("Loading datasets...")
    ds1 = CSVDatasetV2(path="data/finetune/DankDataset.csv",
                       sys_prompt=sys_prompt, tokenizer=tokenizer, max_seq_len=params.max_position_embeddings)
    ds2 = JsonlConversations(path="data/finetune/convos.jsonl", sys_prompt=sys_prompt, tokenizer=tokenizer,
                             max_seq_len=params.max_position_embeddings)
    dataset = torch.utils.data.ConcatDataset(
        [
            ds1, ds2,
            # WizardVicuna(sys_prompt, ),
            HFnoRobotsDataset(sys_prompt, tokenizer, params.max_position_embeddings),
            SmallOrca()

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
          lora_config=lora_params,
          is_lora=False,
          disable_grads_for_embeddings=False,
          disable_scheduler=True,
          learning_rate=1e-4,
          save_every=1000,
          )
