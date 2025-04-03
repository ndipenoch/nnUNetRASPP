import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.network_architecture.generic_UNet import Generic_UNet

class ASPP3D(nn.Module):
    """ Atrous Spatial Pyramid Pooling (ASPP) for 3D Inputs """
    def __init__(self, in_channels, out_channels):
        super(ASPP3D, self).__init__()

        self.conv1x1_1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3x3_2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3x3_3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv1x1_2 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.conv1x1_out = nn.Conv3d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1x1_1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)

        x5 = self.global_avg_pool(x)
        x5 = self.conv1x1_2(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode="trilinear", align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv1x1_out(x)
        return x

class ResidualBlock(nn.Module):
    """ Implements residual connections for UNet encoder blocks """
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)  # Skip connection

class Custom3DUNet(Generic_UNet):
    """ Custom 3D nnU-Net with ASPP at input and residual connections in encoder """
    def __init__(self, input_channels, base_num_features, num_classes, 
                 num_pool, num_conv_per_stage=2, feat_map_mul_on_downscale=2, 
                 conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, dropout_op=nn.Dropout3d, 
                 deep_supervision=True):
        super().__init__(input_channels, base_num_features, num_classes, num_pool, 
                         num_conv_per_stage, feat_map_mul_on_downscale, conv_op, 
                         norm_op, dropout_op, deep_supervision)

        # Replace first convolution layer with ASPP
        self.aspp = ASPP3D(input_channels, base_num_features)

        # Modify encoder blocks to include residual connections
        self._modify_encoder_to_residual()

    def _modify_encoder_to_residual(self):
        """ Wrap each encoder block with residual connections """
        for i in range(len(self.conv_blocks_context)):
            self.conv_blocks_context[i] = ResidualBlock(self.conv_blocks_context[i])

    def forward(self, x):
        x = self.aspp(x)  # Apply ASPP at the input layer
        return super().forward(x)
