import os
import sys
import yaml

def read_config(args):
    if len(args) != 2:
        print("[ERROR] NO CONFIGURATION FILE!")
    else:
        config_path = args[1]

    ##Add Codes for Exception Handling

    print("[SYSTEM] Read {}".format(config_path))

    if config_path is not None:
        with open(config_path) as file:
            config = yaml.safe_load(file)

    return config

def get_device(gpu):
    import torch
    if len(gpu) == 1:
        device = torch.device('cuda:' + str(gpu))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        device = 'cuda'
        print("[SYSTEM] Model is on GPU: {}".format(gpu))

    return device
