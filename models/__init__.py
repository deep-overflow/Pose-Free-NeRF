import torch

## import models below here
from models.posenet.model import PoseNetwork
import torchsummary

## import models up to here

def get_model(configs):
    print('[SYSTEM] Uploading Model...')
    # assign the requested model to variable 'model'
    model = PoseNetwork(
        configs['Encoder_parameters']['num_conv_blocks'],configs['Encoder_parameters']['num_att_blocks'],
        configs['Encoder_parameters']['pos_start_octave'],configs['Decoder_parameters']['num_conv_blocks'],
        configs['Decoder_parameters']['num_att_blocks']
    )
    if model is not None:
        print('[SYSTEM] MODEL SUCCESSFULLY UPLOADED')
        print(model)

    return model