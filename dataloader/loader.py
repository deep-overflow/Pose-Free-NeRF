import torch
from torch.utils.data import DataLoader

def get_dataloader(configs):
    ## wrap train_dataset and test_dataset to dataloader
    if 'Synthetic' in configs['description']:
        from dataloader.data.synthetic_loader import SyntheticLoader

    elif 'coco' in configs['description']:
        from dataloader.data.coco_loader import CoCoLoader