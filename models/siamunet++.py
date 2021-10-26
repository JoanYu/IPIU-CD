# Siamese U-Net++ by Kaiyu Li
# 'SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images'
# IEEE Geoscience and Remote Sensing Letters, 2021. doi: 10.1109/LGRS.2021.3056416.
# https://github.com/likyoo

import torch.nn as nn
import torch

class conv_block_nested(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(conv_block_nested, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.relu(x + residual)
        return output

class upsample(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(upsample, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x
    
class ECAM(nn.Module):
    # Ensemble Channel Attention Module
    # input: B*C*H*W
    # output: B*C*1*1
    
    def __init__(self, in_channels, ratio = 16):
        super(ECAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fullyconv1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fullyconv2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self,x):
        avg_out = self.fullyconv2(self.relu1(self.fullyconv1(self.avg_pool(x))))
        max_out = self.fullyconv2(self.relu1(self.fullyconv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    net=CAM2(512,10)
    print(net(input).shape)
    print('okk!')