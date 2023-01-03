import torch
from torch.utils.data import DataLoader


def get_dataloader(configs):
    ## wrap train_dataset and test_dataset to dataloader
    if 'Blender' in configs['description']:
        from dataloader.data.blender_loader import blenderDataset
        path = configs['path']
        scene = configs['scene']
        train_dataset = blenderDataset(path, scene, mode='train')
        test_dataset = blenderDataset(path,scene,mode='test')

    '''
    elif 'coco' in configs['description']:
        from dataloader.data.coco_loader import CoCoLoader
    '''

    # Wrap torch.Dataset to torch.DataLoader
    data_loader = dict()
    train_batch = configs['batch']
    test_batch = configs['batch']

    train_loader = DataLoader(
        train_dataset, train_batch, shuffle=True,
        drop_last=True, pin_memory=True,
        num_workers=configs['num_workers']
    )
    test_loader = DataLoader(
        test_dataset, test_batch, shuffle=False,
        drop_last=True, pin_memory=True,
        num_workers=configs['num_workers']
    )
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader

    return data_loader
