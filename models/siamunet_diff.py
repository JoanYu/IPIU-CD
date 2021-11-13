# Siamese U-Net (diff) by Rodrigo Caye Daudt
# "Fully convolutional siamese networks for change detection" ICIP 2018 arxiv: 1810.08462
# https://rcdaudt.github.io/

import torch
from torch.functional import block_diag
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d


def maxpool(planes):
    return F.max_pool2d(planes, kernel_size=2, stride=2)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv3 = False):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(False)
        self.dropout = nn.Dropout2d(0.2)
        self.with_conv3 = with_conv3
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.with_conv3:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, with_conv2 = False):
        super(DecoderBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.deconv2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.deconv3 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(False)
        self.dropout = nn.Dropout2d(0.2)
        self.with_conv2 = with_conv2
    
    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.with_conv2:
            x = self.deconv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
        

class SiamUnet_diff(nn.Module):
    # SiamUnet_diff Segmentation Network.

    def __init__(self, in_channels, out_channels):
        super(SiamUnet_diff, self).__init__()

        self.num_channels = [16, 32, 64, 128, 256]
        # self.self.num_channels = [8, 16, 32, 64, 128]
        
        self.softmax = nn.LogSoftmax(dim=1)

        self.encoder1 = EncoderBlock(in_channels, self.num_channels[0])
        self.encoder2 = EncoderBlock(self.num_channels[0], self.num_channels[1])
        self.encoder3 = EncoderBlock(self.num_channels[1], self.num_channels[2], with_conv3=True)
        self.encoder4 = EncoderBlock(self.num_channels[2], self.num_channels[3], with_conv3=True)
        self.upconv4 = nn.ConvTranspose2d(self.num_channels[3], self.num_channels[3], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.decoder4 = DecoderBlock(self.num_channels[4], self.num_channels[3], self.num_channels[2], with_conv2=True)
        self.upconv3 = nn.ConvTranspose2d(self.num_channels[2], self.num_channels[2], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.decoder3 = DecoderBlock(self.num_channels[3], self.num_channels[2], self.num_channels[1], with_conv2=True)
        self.upconv2 = nn.ConvTranspose2d(self.num_channels[1], self.num_channels[1], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.decoder2 = DecoderBlock(self.num_channels[2], self.num_channels[1], self.num_channels[0])
        self.upconv1 = nn.ConvTranspose2d(self.num_channels[0], self.num_channels[0], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.decoder1 = DecoderBlock(self.num_channels[1], self.num_channels[0], out_channels)

        
    def forward(self, x1, x2):
        # Encoding
        # Stage 1
        x1_1 = self.encoder1(x1)
        x1p_1 = maxpool(x1_1)
        x1_2 = self.encoder1(x2)
        x1p_2 = maxpool(x1_2)

        # Stage 2
        x2_1 = self.encoder2(x1p_1)
        x2p_1 = maxpool(x2_1)
        x2_2 = self.encoder2(x1p_2)
        x2p_2 = maxpool(x2_2)

        # Stage 3
        x3_1 = self.encoder3(x2p_1)
        x3p_1 = maxpool(x3_1)
        x3_2 = self.encoder3(x2p_2)
        x3p_2 = maxpool(x3_2)

        # Stage 4
        x4_1 = self.encoder4(x3p_1)
        # x4p_1 = self.maxpool(x4_1)
        x4_2 = self.encoder4(x3p_2)
        x4p_2 = maxpool(x4_2)

        # Decoding
        # Stage 4
        x4d = self.upconv4(x4p_2)
        pad4 = ReplicationPad2d((0, x4_1.size(3) - x4d.size(3), 0, x4_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x4_1 - x4_2)), 1)
        out4 = self.decoder4(x4d)

        # Stage 3
        x3d = self.upconv3(out4)
        pad3 = ReplicationPad2d((0, x3_1.size(3) - x3d.size(3), 0, x3_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x3_1 - x3_2)), 1)
        out3 = self.decoder3(x3d)
        
        # Stage 2
        x2d = self.upconv2(out3)
        pad2 = ReplicationPad2d((0, x2_1.size(3) - x2d.size(3), 0, x2_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x2_1 - x2_2)), 1)
        out2 = self.decoder2(x2d)

        # Stage 1
        x1d = self.upconv1(out2)
        pad1 = ReplicationPad2d((0, x1_1.size(3) - x1d.size(3), 0, x1_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x1_1 - x1_2)), 1)
        out1 = self.decoder1(x1d)

        return self.softmax(out1)

if __name__ == '__main__':
    input1=torch.randn(16,3,256,256).cuda()
    input2=torch.randn(16,3,256,256).cuda()
    siamunet_dif=SiamUnet_diff(3,10).cuda()
    print(siamunet_dif(input1, input2).shape)
    print('okk!')