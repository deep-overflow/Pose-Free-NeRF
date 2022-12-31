import torch
import os
import time
import pandas as ps

# When saving model, Save the epoch information together.
def model_save(configs,model,epoch):
    print("Saved Epoch: {}".format(epoch))
    path = "./model_save/{}".format(configs["id"])
    check_directory("./model_save")
    check_directory(path)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }, path + "/best_model_" + str(configs["seed"])
    )


def check_directory(directory_path):
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
