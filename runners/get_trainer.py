def get_trainer(dl,model,configs):
    # return the trainer following the requests in configuration
    if configs['description'] =='PoseNet':
        from runners.trainer_posenet import Trainer
        trainer = Trainer(dl,model,configs)

    return trainer
