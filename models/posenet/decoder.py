import numpy as np
import torch
import torch.nn as nn
from models.posenet.layers import DecoderTransformer

import math


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class PoseNet(nn.Module):
    def __init__(self,num_conv_blocks = 4, num_att_blocks =3):
        super().__init__()

        conv_blocks = [SRTConvBlock(idim = 3,hdim=96)]
        cur_hdim = 192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.per_patch_linear = nn.Conv2d(cur_hdim, 768, kernel_size=1)
        self.transformer = DecoderTransformer(768, depth=num_att_blocks, heads=12, dim_head=64,
                                       mlp_dim=1536, kv_dim=768)
        self.pose_estimate_mlp = nn.Linear(768,12)

    def forward(self,SLSR,images):
        """
        Args:
            SLSR : [batch_size, num_images*num_patches_per_image,channels_per_patch] Set-Latent Scene Representation. Output of Encoder.
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
        Returns:
            Estimated Pose: [batch_size, num_patches, channels_per_patch]
        """

        batch_size,num_images = images.shape[:2]

        x = images.flatten(0,1)
        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        x = x.flatten(2,3).permute(0,2,1)

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)

        x = self.transformer(x,z=SLSR)

        #Apply average pooling to patches that belong to the same image.
        result = torch.zeros(batch_size, num_images, channels_per_patch)
        for i in range(0,batch_size):
            for j in range(0,num_images):
                tmp = x[i,j*patches_per_image:(j+1)*patches_per_image,:]
                result[i,j,:] = torch.mean(tmp,0)

        result = self.pose_estimate_mlp(result).view(batch_size,num_images,3,4)
        return result
