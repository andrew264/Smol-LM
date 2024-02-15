from safetensors import safe_open
import torch
import os

from safetensors.torch import save_file
from torch.utils.data import DataLoader

from model import ModelConfig, Transformer
from main import validate_model, NPDataset

if __name__ == '__main__':
    model_checkpoint = './weights/accelerator_states/model.safetensors'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    NUM_LAYERS_TO_REMOVE = 4

    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint}")

    if os.path.exists('../weights/config.json'):
        params = ModelConfig.from_json('../weights/config.json')
        print("Loaded config from file.")
    else:
        raise FileNotFoundError("Config file not found.")

    state_dict = {}
    d = device.type if device.type == 'cpu' else device.index
    with safe_open(model_checkpoint, framework="pt", device=d) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    num_params_before = sum([p.numel() for p in state_dict.values()])

    config = ModelConfig.from_json('../weights/config.json')

    if config.num_hidden_layers - NUM_LAYERS_TO_REMOVE < 1:
        raise ValueError("Model must have at least one layer.")

    # remove layers from middle of model
    middle_idx = config.num_hidden_layers // 2
    start_idx = middle_idx - (NUM_LAYERS_TO_REMOVE // 2)
    end_idx = start_idx + NUM_LAYERS_TO_REMOVE
    idx_to_remove = list(range(start_idx, end_idx))

    state_dict_keys = list(state_dict.keys())
    for k in state_dict_keys:
        if any([f'layers.{i}.' in k for i in idx_to_remove]):
            print(f"Removing layer {k}")
            state_dict.pop(k)

    config.num_hidden_layers -= NUM_LAYERS_TO_REMOVE

    model = Transformer(config)
    model.to(dtype=torch.bfloat16, device=device)

    renumbered_state_dict = {}
    for key in model.state_dict().keys():
        if key not in state_dict:
            layer_num = int(key.split('.')[1])
            print(f"Renumbering layer.{layer_num + NUM_LAYERS_TO_REMOVE} to layer.{layer_num}")
            new_key = key.replace(f'.{layer_num}.', f'.{layer_num + NUM_LAYERS_TO_REMOVE}.')
            renumbered_state_dict[key] = state_dict[new_key]
            state_dict.pop(new_key)
            print(f"Removed layer {new_key} from state_dict.")
        else:
            renumbered_state_dict[key] = state_dict[key]

    num_params_after = sum([p.numel() for p in renumbered_state_dict.values()])

    model.load_state_dict(renumbered_state_dict)
    model.eval()
    dataset = NPDataset('../data/processed/val.bin', params.max_position_embeddings)
    val_data = DataLoader(dataset, batch_size=params.max_batch_size * 4, shuffle=False, drop_last=True, pin_memory=True)

    print(f"Number of parameters before: {num_params_before / 1e6:.2f}M parameters.")
    print(f"Number of parameters after: {num_params_after / 1e6:.2f}M parameters.")
    print(f"Number of parameters removed: {(num_params_before - num_params_after) / 1e6:.2f}M parameters.")
    print(f"Percentage of parameters removed: {(num_params_before - num_params_after) / num_params_before * 100:.2f}%")

    validate_model(model, val_data, full_validation=False)
    print("Validation complete.")

    # Save the model
    save_file(model.state_dict(), model_checkpoint)
    print(f"Model saved to {model_checkpoint}")
    config.to_json('./weights/config.json')
    print(f"Config saved to ./weights/config.json")
