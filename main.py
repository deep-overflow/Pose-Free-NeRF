### Follow the formats below
    # python main.py ( your configuration file - typically .yaml files stored in configs )

import wandb
import os
import sys

from models import get_model
from wandb_utils import start_wandb
from utils.system_utils import *
from dataloader.loader import get_dataloader
from runners.get_trainer import get_trainer

configs = read_config(sys.argv)
dl = get_dataloader(configs['dataset'])
model = get_model(configs['model'])
# Delete # to use wandb
# start_wandb.set_wandb(model,configs['wandb'],configs['train'],configs['dataset'])
trainer = get_trainer(dl=dl,model=model,configs=configs['train'])
trainer.run()

