import torch
from rustposenet import RUSTPoseNet
from utils import Config

args_dict = {
    'img_h': 256,
    'img_w': 256,
    'embed_dim': 1024,
    'num_encoder_blocks': 3,
    'num_multi_heads': 8,
}

configs = Config(args_dict)

posenet = RUSTPoseNet(configs)

inputs = torch.ones((1, 3, configs.img_h, configs.img_w // 2))
SLSRs = torch.ones((5, configs.embed_dim, configs.img_h // 8, configs.img_w // 8))
SLSRs = SLSRs.reshape(1, 5 * (configs.img_h // 8) * (configs.img_w // 8), configs.embed_dim)

outputs = posenet(inputs, SLSRs)

print(outputs.shape)