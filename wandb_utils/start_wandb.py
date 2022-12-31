import wandb


def set_wandb(model, config, train_config, ds_config):
    wandb.init(project=config['project'],
               entity=config['entity'],
               name=train_config['id'])

    config = wandb.config
    config.lr = train_config['Learning_Rate']
    config.batch = ds_config['batch']
    wandb.watch(model)
