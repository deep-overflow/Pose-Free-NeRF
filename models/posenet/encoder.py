import numpy as np
import torch
import torch.nn as nn
from models.posenet.layers import RayEncoder, Transformer

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


class ImprovedSRTEncoder(nn.Module):
    """
    Scene Representation Transformer Encoder with the improvements from Appendix A.4 in the OSRT paper.
    """

    def __init__(self, num_conv_blocks=3, num_att_blocks=5, pos_start_octave=0, image_size=800, pose_embedding=True):
        super().__init__()
        if pose_embedding:
            self.ray_encoder = RayEncoder(pos_octaves=15, pos_start_octave=pos_start_octave,
                                          ray_octaves=15)
            conv_blocks = [SRTConvBlock(idim=183, hdim=96)]
        else:
            self.ray_encoder = None
            conv_blocks = [SRTConvBlock(idim=3, hdim=96)]

        cur_hdim = 192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.per_patch_linear = nn.Conv2d(cur_hdim, 768, kernel_size=1)

        feature_map_size = int(image_size / 2 ** num_conv_blocks)
        # Original SRT initializes with stddev=1/math.sqrt(d).
        embedding_stdev = (1. / math.sqrt(768))
        self.pixel_embedding = nn.Parameter(torch.randn(1, 768, feature_map_size, feature_map_size) * embedding_stdev)
        self.first_image_embedding = nn.Parameter(torch.randn(1, 1, 768) * embedding_stdev)
        self.non_first_image_embedding = nn.Parameter(torch.randn(1, 1, 768) * embedding_stdev)

        self.transformer = Transformer(768, depth=num_att_blocks, heads=12, dim_head=64,
                                       mlp_dim=1536, selfatt=True)

    def forward(self, images, camera_pos, rays):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        image_idxs = torch.zeros(batch_size, num_images)
        image_idxs[:, 0] = 1
        image_idxs = image_idxs.flatten(0, 1).unsqueeze(-1).unsqueeze(-1).to(x)
        image_embedding = image_idxs * self.first_image_embedding + \
                          (1. - image_idxs) * self.non_first_image_embedding

        if self.ray_encoder is not None:
            ray_enc = self.ray_encoder(camera_pos, rays)
            x = torch.cat((x, ray_enc), 1)

        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        x = x + self.pixel_embedding
        x = x.flatten(2, 3).permute(0, 2, 1)
        x = x + image_embedding

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)

        x = self.transformer(x)

        return x
