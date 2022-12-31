import torch


class CoCoLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, configs, split='train'):
        data_path = configs['path']

    def __len__(self):

    def __getitem__(self, idx):