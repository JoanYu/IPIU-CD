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
        
        self.softmax = nn.LogSoftmax(dim=1)

    def conv(self, in_planes, out_planes):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
    def batchnorm(self, planes):
        return nn.BatchNorm2d(planes)
    def maxpool(self, planes):
        return F.max_pool2d(planes, kernel_size=2, stride=2)
    def upconv(self, in_planes, out_planes):
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, padding=1, stride=2, output_padding=1)
    def deconv(self, in_planes, out_planes):
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, padding=1)

    def _make_encoder_layers(self):
        encoder_layers = []
        for i in range(4):
            encoder_layer=[]
            conv1 = self.conv(self.in_channels, self.num_channels[i]) if i == 0 else \
                self.conv(self.num_channels[i-1], self.num_channels[i])
            bn1 = self.batchnorm(self.num_channels[i])
            conv2 = self.conv(self.num_channels[i], self.num_channels[i])
            bn2 = self.batchnorm(self.num_channels[i])
            relu = nn.ReLU()
            dropout = nn.Dropout2d(p=0.2)
            encoder_layer.extend([conv1,bn1,relu,dropout,conv2,bn2,relu,dropout])
            if i >= 2: 
                conv3 = self.conv(self.num_channels[i], self.num_channels[i])
                bn3 = self.batchnorm(self.num_channels[i])
                encoder_layer.extend([conv3,bn3,relu,dropout])
            encoder_layers.append(nn.Sequential(*encoder_layer))
        return encoder_layers
    
    def _make_decoder_layers(self):
        decoder_layers = []
        for i in range(4):
            decoder_layer=[]
            deconv1 = self.deconv(self.num_channels[i+1], self.num_channels[i])
            bn1 = self.batchnorm(self.num_channels[i])
            deconv3 = self.conv(self.num_channels[i], self.out_channels) if i == 0 else \
                self.conv(self.num_channels[i], self.num_channels[i-1])
            bn3 = self.batchnorm(self.out_channels) if i == 0 else self.batchnorm(self.num_channels[i-1])
            relu = nn.ReLU()
            dropout = nn.Dropout2d(p=0.2)
            if i >= 2: 
                deconv2 = self.conv(self.num_channels[i], self.num_channels[i])
                bn2 = self.batchnorm(self.num_channels[i])
                decoder_layer.extend([deconv1,bn1,relu,dropout,deconv2,bn2,relu,dropout,deconv3,bn3,relu,dropout])
            else: decoder_layer.extend([deconv1,bn1,relu,dropout,deconv3,bn3,relu,dropout])
            decoder_layers.append(nn.Sequential(*decoder_layer))
        return decoder_layers

    def forward(self, x1, x2):
        # Encoding
        encoder_layers = self._make_encoder_layers()

        # Stage 1
        x1_1 = encoder_layers[0](x1)
        x1p_1 = self.maxpool(x1_1)
        x1_2 = encoder_layers[0](x2)
        x1p_2 = self.maxpool(x1_2)

        # Stage 2
        x2_1 = encoder_layers[1](x1p_1)
        x2p_1 = self.maxpool(x2_1)
        x2_2 = encoder_layers[1](x1p_2)
        x2p_2 = self.maxpool(x2_2)

        # Stage 3
        x3_1 = encoder_layers[2](x2p_1)
        x3p_1 = self.maxpool(x3_1)
        x3_2 = encoder_layers[2](x2p_2)
        x3p_2 = self.maxpool(x3_2)

        # Stage 4
        x4_1 = encoder_layers[3](x3p_1)
        # x4p_1 = self.maxpool(x4_1)
        x4_2 = encoder_layers[3](x3p_2)
        x4p_2 = self.maxpool(x4_2)

        # Decoding
        decoder_layers = self._make_decoder_layers()

        # Stage 4
        x4d = self.upconv(self.num_channels[3], self.num_channels[3])(x4p_2)
        pad4 = ReplicationPad2d((0, x4_1.size(3) - x4d.size(3), 0, x4_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x4_1 - x4_2)), 1)
        out4 = decoder_layers[3](x4d)

        # Stage 3
        x3d = self.upconv(self.num_channels[2], self.num_channels[2])(out4)
        pad3 = ReplicationPad2d((0, x3_1.size(3) - x3d.size(3), 0, x3_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x3_1 - x3_2)), 1)
        out3 = decoder_layers[2](x3d)
        
        # Stage 2
        x2d = self.upconv(self.num_channels[1], self.num_channels[1])(out3)
        pad2 = ReplicationPad2d((0, x2_1.size(3) - x2d.size(3), 0, x2_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x2_1 - x2_2)), 1)
        out2 = decoder_layers[1](x2d)

        # Stage 1
        x1d = self.upconv(self.num_channels[0], self.num_channels[0])(out2)
        pad1 = ReplicationPad2d((0, x1_1.size(3) - x1d.size(3), 0, x1_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x1_1 - x1_2)), 1)
        out1 = decoder_layers[0](x1d)

        return self.softmax(out1)

if __name__ == '__main__':
    input1=torch.randn(16,3,256,256)
    input2=torch.randn(16,3,256,256)
    siamunet_dif=SiamUnet_diff(3,10)
    print(siamunet_dif(input1, input2).shape)
    print('okk!')