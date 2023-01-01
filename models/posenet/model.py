from torch import nn

from models.posenet.encoder import ImprovedSRTEncoder
from models.posenet.decoder import PoseNet

class PoseNetwork(nn.Module):
    def __init__(self,encoder_conv_blocks,encoder_att_blocks,encoder_pos_start_octave,encoder_image_size,encoder_pose_embedding,decoder_conv_blocks,decoder_att_blocks):
        super(PoseNetwork,self).__init__()
        self.encoder = ImprovedSRTEncoder(encoder_conv_blocks,encoder_att_blocks,encoder_pos_start_octave,
                                          encoder_image_size,encoder_pose_embedding)
        self.decoder = PoseNet(decoder_conv_blocks,decoder_att_blocks)

    def forward(self,encoder_input,decoder_input):
        SLSR = self.encoder(encoder_input)
        estimated_pose = self.decoder(SLSR,decoder_input)

        return estimated_pose