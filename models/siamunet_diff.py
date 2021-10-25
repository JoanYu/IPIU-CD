# Siamese U-Net (diff) by Rodrigo Caye Daudt
# "Fully convolutional siamese networks for change detection" ICIP 2018 arxiv: 1810.08462
# https://rcdaudt.github.io/

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.padding import ReplicationPad2d

class SiamUnet_diff(nn.Module):
    # SiamUnet_diff Segmentation Network.

    def __init__(self, in_channels, out_channels):
        super(SiamUnet_diff, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = [16, 32, 64, 128, 256]
        # self.num_channels = [8, 16, 32, 64, 128]

    def conv(self, in_planes, out_planes):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
    
    def batchnorm(self, planes):
        return nn.BatchNorm2d(planes)
    
    def dropout(self):
        return nn.Dropout2d(p=0.2)
    
    def maxpool(self, planes):
        return F.max_pool2d(planes, kernel_size=2, stride=2)
    
    def upconv(self, in_planes, out_planes):
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, padding=1, stride=2, output_padding=1)
    
    def deconv(self, in_planes, out_planes):
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, padding=1)
    
    def softmax(self):
        return nn.LogSoftmax(dim=1)
        
    def forward(self, x1, x2):
        # Encoding
        # for x1
        # Stage 1
        x11 = self.conv(self.in_channels, self.num_channels[0])(x1)
        x11 = self.batchnorm(self.num_channels[0])(x11)
        x11 = self.dropout()(F.relu(x11))

        x12_1 = self.conv(self.num_channels[0], self.num_channels[0])(x11)
        x12_1 = self.batchnorm(self.num_channels[0])(x12_1)
        x12_1 = self.dropout()(F.relu(x12_1))

        x1p_1 = self.maxpool(x12_1)

        # Stage 2
        x21 = self.conv(self.num_channels[0], self.num_channels[1])(x1p_1)
        x21 = self.batchnorm(self.num_channels[1])(x21)
        x21 = self.dropout()(F.relu(x21))

        x22_1 = self.conv(self.num_channels[1], self.num_channels[1])(x21)
        x22_1 = self.batchnorm(self.num_channels[1])(x22_1)
        x22_1 = self.dropout()(F.relu(x22_1))

        x2p_1 = self.maxpool(x22_1)

        # Stage 3
        x31 = self.conv(self.num_channels[1], self.num_channels[2])(x2p_1)
        x31 = self.batchnorm(self.num_channels[2])(x31)
        x31 = self.dropout()(F.relu(x31))

        x32 = self.conv(self.num_channels[2], self.num_channels[2])(x31)
        x32 = self.batchnorm(self.num_channels[2])(x32)
        x32 = self.dropout()(F.relu(x32))

        x33_1 = self.conv(self.num_channels[2], self.num_channels[2])(x32)
        x33_1 = self.batchnorm(self.num_channels[2])(x33_1)
        x33_1 = self.dropout()(F.relu(x33_1))

        x3p_1 = self.maxpool(x33_1)

        # Stage 4
        x41 = self.conv(self.num_channels[2], self.num_channels[3])(x3p_1)
        x41 = self.batchnorm(self.num_channels[3])(x41)
        x41 = self.dropout()(F.relu(x41))

        x42 = self.conv(self.num_channels[3], self.num_channels[3])(x41)
        x42 = self.batchnorm(self.num_channels[3])(x42)
        x42 = self.dropout()(F.relu(x42))

        x43_1 = self.conv(self.num_channels[3], self.num_channels[3])(x42)
        x43_1 = self.batchnorm(self.num_channels[3])(x43_1)
        x43_1 = self.dropout()(F.relu(x43_1))

        # x4p_1 = self.maxpool(x43_1)

        # for x2
        # Stage 1
        x11 = self.conv(self.in_channels, self.num_channels[0])(x2)
        x11 = self.batchnorm(self.num_channels[0])(x11)
        x11 = self.dropout()(F.relu(x11))

        x12_2 = self.conv(self.num_channels[0], self.num_channels[0])(x11)
        x12_2 = self.batchnorm(self.num_channels[0])(x12_2)
        x12_2 = self.dropout()(F.relu(x12_2))

        x1p_2 = self.maxpool(x12_2)

        # Stage 2
        x21 = self.conv(self.num_channels[0], self.num_channels[1])(x1p_2)
        x21 = self.batchnorm(self.num_channels[1])(x21)
        x21 = self.dropout()(F.relu(x21))

        x22_2 = self.conv(self.num_channels[1], self.num_channels[1])(x21)
        x22_2 = self.batchnorm(self.num_channels[1])(x22_2)
        x22_2 = self.dropout()(F.relu(x22_2))

        x2p_2 = self.maxpool(x22_2)

        # Stage 3
        x31 = self.conv(self.num_channels[1], self.num_channels[2])(x2p_2)
        x31 = self.batchnorm(self.num_channels[2])(x31)
        x31 = self.dropout()(F.relu(x31))

        x32 = self.conv(self.num_channels[2], self.num_channels[2])(x31)
        x32 = self.batchnorm(self.num_channels[2])(x32)
        x32 = self.dropout()(F.relu(x32))

        x33_2 = self.conv(self.num_channels[2], self.num_channels[2])(x32)
        x33_2 = self.batchnorm(self.num_channels[2])(x33_2)
        x33_2 = self.dropout()(F.relu(x33_2))

        x3p_2 = self.maxpool(x33_2)

        # Stage 4
        x41 = self.conv(self.num_channels[2], self.num_channels[3])(x3p_2)
        x41 = self.batchnorm(self.num_channels[3])(x41)
        x41 = self.dropout()(F.relu(x41))

        x42 = self.conv(self.num_channels[3], self.num_channels[3])(x41)
        x42 = self.batchnorm(self.num_channels[3])(x42)
        x42 = self.dropout()(F.relu(x42))

        x43_2 = self.conv(self.num_channels[3], self.num_channels[3])(x42)
        x43_2 = self.batchnorm(self.num_channels[3])(x43_2)
        x43_2 = self.dropout()(F.relu(x43_2))

        # x4p_2 = self.maxpool(x43_2)
        x4p = self.maxpool(x43_2)

        # Decoding
        # Stage 4
        x4d = self.upconv(self.num_channels[3], self.num_channels[3])(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)

        x43d = self.deconv(self.num_channels[4], self.num_channels[3])(x4d)
        x43d = self.batchnorm(self.num_channels[3])(x43d)
        x43d = self.dropout()(F.relu(x43d))

        x42d = self.deconv(self.num_channels[3], self.num_channels[3])(x43d)
        x42d = self.batchnorm(self.num_channels[3])(x42d)
        x42d = self.dropout()(F.relu(x42d))

        x41d = self.deconv(self.num_channels[3], self.num_channels[2])(x42d)
        x41d = self.batchnorm(self.num_channels[2])(x41d)
        x41d = self.dropout()(F.relu(x41d))

        # Stage 3
        x3d = self.upconv(self.num_channels[2], self.num_channels[2])(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)

        x33d = self.deconv(self.num_channels[3], self.num_channels[2])(x3d)
        x33d = self.batchnorm(self.num_channels[2])(x33d)
        x33d = self.dropout()(F.relu(x33d))

        x32d = self.deconv(self.num_channels[2], self.num_channels[2])(x33d)
        x32d = self.batchnorm(self.num_channels[2])(x32d)
        x32d = self.dropout()(F.relu(x32d))

        x31d = self.deconv(self.num_channels[2], self.num_channels[1])(x32d)
        x31d = self.batchnorm(self.num_channels[1])(x31d)
        x31d = self.dropout()(F.relu(x31d))

        # Stage 2
        x2d = self.upconv(self.num_channels[1], self.num_channels[1])(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)

        x22d = self.deconv(self.num_channels[2], self.num_channels[1])(x2d)
        x22d = self.batchnorm(self.num_channels[1])(x22d)
        x22d = self.dropout()(F.relu(x22d))

        x21d = self.deconv(self.num_channels[1], self.num_channels[0])(x22d)
        x21d = self.batchnorm(self.num_channels[0])(x21d)
        x21d = self.dropout()(F.relu(x21d))

        # Stage 1
        x1d = self.upconv(self.num_channels[0], self.num_channels[0])(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)

        x12d = self.deconv(self.num_channels[1], self.num_channels[0])(x1d)
        x12d = self.batchnorm(self.num_channels[0])(x12d)
        x12d = self.dropout()(F.relu(x12d))

        x11d = self.deconv(self.num_channels[0], self.out_channels)(x12d)
        x11d = self.batchnorm(self.out_channels)(x11d)
        x11d = self.dropout()(F.relu(x11d))

        return self.softmax()(x11d)

if __name__ == '__main__':
    input1=torch.randn(16,3,256,256)
    input2=torch.randn(16,3,256,256)
    siamunet_dif=SiamUnet_diff(3,10)
    print(siamunet_dif(input1, input2).shape)
    print('okk!')