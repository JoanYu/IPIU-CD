# Siamese U-Net++(ECAM) by Kaiyu Li
# 'SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images'
# IEEE Geoscience and Remote Sensing Letters, 2021. doi: 10.1145/3437802.3437810
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

class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

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
    def __init__(self, in_channels, ratio=16):
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
    def __init__(self, in_planes=3, out_planes=2, init_feature = 32):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        filters = [init_feature, init_feature * 2, init_feature * 4, init_feature * 8, init_feature * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convs = []
        self.Ups = []
        for j in range(4):
            self.conv = [conv_block_nested(in_planes, filters[0], filters[0])] if j == 0 else \
                [conv_block_nested(filters[0] * (j+1) + filters[1], filters[0], filters[0])]
            self.Up = []
            for i in range(4-j):
                tmp_conv = conv_block_nested(filters[i], filters[i+1], filters[i+1]) if j == 0 else \
                    conv_block_nested(filters[i+1] * (j+1) + filters[i+2], filters[i+1], filters[i+1])
                self.conv.append(tmp_conv)
                self.Up.append(up(filters[i+1]))
            self.convs.append(self.conv)
            self.Ups.append(self.Up)
        self.conv4 = [conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])]
        self.convs.append(self.conv4)

        self.ca = ECAM(filters[0] * 4, ratio=16)
        self.ca1 = ECAM(filters[0], ratio=16 // 4)

        self.conv_final = nn.Conv2d(filters[0] * 4, out_planes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, xA, xB):
        x0_0A = self.convs[0][0](xA)
        x1_0A = self.convs[0][1](self.pool(x0_0A))
        x2_0A = self.convs[0][2](self.pool(x1_0A))
        x3_0A = self.convs[0][3](self.pool(x2_0A))

        x0_0B = self.convs[0][0](xB)
        x1_0B = self.convs[0][1](self.pool(x0_0B))
        x2_0B = self.convs[0][2](self.pool(x1_0B))
        x3_0B = self.convs[0][3](self.pool(x2_0B))
        x4_0B = self.convs[0][4](self.pool(x3_0B))

        x0_1 = self.convs[1][0](torch.cat([x0_0A, x0_0B, self.Ups[0][0](x1_0B)], 1))
        x1_1 = self.convs[1][1](torch.cat([x1_0A, x1_0B, self.Ups[0][1](x2_0B)], 1))
        x0_2 = self.convs[2][0](torch.cat([x0_0A, x0_0B, x0_1, self.Ups[1][0](x1_1)], 1))


        x2_1 = self.convs[1][2](torch.cat([x2_0A, x2_0B, self.Ups[0][2](x3_0B)], 1))
        x1_2 = self.convs[2][1](torch.cat([x1_0A, x1_0B, x1_1, self.Ups[1][1](x2_1)], 1))
        x0_3 = self.convs[3][0](torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Ups[2][0](x1_2)], 1))

        x3_1 = self.convs[1][3](torch.cat([x3_0A, x3_0B, self.Ups[0][3](x4_0B)], 1))
        x2_2 = self.convs[2][2](torch.cat([x2_0A, x2_0B, x2_1, self.Ups[1][2](x3_1)], 1))
        x1_3 = self.convs[3][1](torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Ups[2][1](x2_2)], 1))
        x0_4 = self.convs[4][0](torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Ups[3][0](x1_3)], 1))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)

        return out
            
if __name__ == '__main__':
    input1=torch.randn(16,4,256,256)
    input2=torch.randn(16,4,256,256)
    net=SNUNet_ECAM(4,6)
    print(net(input1,input2).shape)
    print('okk!')