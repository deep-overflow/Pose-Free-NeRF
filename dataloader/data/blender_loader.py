import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import re
import copy
import numpy as np
import json
import os

import imageio
import numpy as np
import torch
from einops import repeat, rearrange


class blenderDataset(Dataset):
    def __init__(self,
                 datadir: str,
                 scene_name: str,
                 train_skip: int = 1,
                 val_skip: int = 1,
                 test_skip: int = 1,
                 cam_scale_factor: float = 1.0,
                 white_bkgd: bool = True,
                 mode: str = 'train',  # 3 Options : 'train', 'test' 'val'
                 ):
        basedir = os.path.join(datadir, scene_name)
        with open(os.path.join(basedir, "transforms_{}.json".format(mode)), "r") as fp:
            meta = json.load(fp)

        # # GT로 설정할 view number를 하드코딩으로 지정합니다.
        # Pose Network를 훈련하는 동안은, (img , pose) 이렇게 하나씩만 return하게 합니다.
        views_gt = []#[0, 1, 2]

        p = re.compile("(\w|.)*\/\w*\/r_(\d*)")

        self.imgs = []
        self.poses = []
        self.imgs_gt = []
        self.poses_gt = []

        if mode == "train":
            skip = train_skip
        elif mode == "val":
            skip = val_skip
        elif mode == "test":
            skip = test_skip

        for frame in meta["frames"][::skip]:
            m = p.match(frame["file_path"])
            cur_view = int(m.group(2))

            if (cur_view not in views_gt):
                fname = os.path.join(basedir, frame["file_path"] + ".png")
                self.imgs.append(imageio.v2.imread(fname))
                self.poses.append(np.array(frame["transform_matrix"]))

            else:
                fname = os.path.join(basedir, frame["file_path"] + ".png")
                self.imgs_gt.append(imageio.v2.imread(fname))
                self.poses_gt.append(np.array(frame["transform_matrix"]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # imgs[0] ~ imgs[2] : GT
        # imgs [3] : target

        # Return
        # imgs : torch.Tensor (4, 4, 800, 800) - (bs, C, H, W)
        # poses : torch.Tensor [4, 4, 4]

        # GT 담긴 리스트들을 먼저 불러온다
        imgs = copy.deepcopy(self.imgs_gt)
        poses = copy.deepcopy(self.poses_gt)

        # idx에 해당하는 이미지와 포즈를 넣는다
        imgs.append(self.imgs[idx])
        poses.append(self.poses[idx])

        # Normalize
        imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)

        # Conver to torch.Tensor
        imgs = torch.as_tensor(imgs)
        imgs = rearrange(imgs, 'bs h w c -> bs c h w').flatten(0,1)
        # 현재 SRT Encoder는 RGB channel만 받기 때문에, 3 채널만 가져갑니다.
        imgs = imgs[:3,:,:]
        poses = torch.as_tensor(poses).flatten(0,1)

        return imgs, poses