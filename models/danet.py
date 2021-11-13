# DANet by CASIA IVA
# "Dual Attention Network for Scene Segmentation" arxiv: 1809.02983
# jliu@nlpr.ia.ac.cn

import numpy as np
import torch
from torch import nn
from torch.nn import init

class PAM(nn.Module):
    # Ref from SAGAN
    '''
    Position Attention Module
    '''

    def __init__(self,in_planes, ratio=8):
        super(PAM, self).__init__()
        self.in_planes = in_planes
        self.key_conv=nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1)
        self.value_conv=nn.Conv2d(in_planes, in_planes ,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            x: B*C*H*W
            q: B*(C/r)*(H*W)
            v: B*(C/r)*(H*W)
            attention: B*(H*W)*(H*W)
            out: B*C*H*W
        """
        B, C, H, W = x.size()
        query = self.key_conv(x).view(B, -1, H*W)
        key = self.key_conv(x).view(B, -1, H*W)
        attention = self.softmax(torch.bmm(query.permute(0, 2, 1), key))
        value = self.value_conv(x).view(B, -1, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)
        out = self.gamma*out + x
        return out

class CAM(nn.Module):
    '''
    Channel Attention Module
    '''

    def __init__(self,in_planes, ratio=8):
        super(CAM, self).__init__()
        self.in_planes = in_planes
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            x/out: B*C*H*W
            q/k/v: B*C*(H*W)
            attention: B*C*C
        """
        B, C, H, W = x.size()
        query = x.view(B, C, -1)
        key = x.view(B, C, -1)
        energy = torch.bmm(query, key.permute(0, 2, 1))
        attention = self.softmax(torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy)
        value = x.view(B, C, -1)

        out = torch.bmm(attention, value).view(B, C, H, W)
        out = self.gamma*out + x
        return out
    
class DAM(nn.Module):
    # Dual Attention Module
    # 
    def __init__(self, in_channels, out_channels, ratio=4):
        super(DAM, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels//ratio, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels//ratio),
                                    nn.ReLU())

        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels//ratio, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels//ratio),
                                    nn.ReLU())
        
        self.pa = PAM(in_channels//ratio)
        self.ca = CAM(in_channels//ratio)
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels//ratio, in_channels//ratio, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels//ratio),
                                    nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels//ratio, in_channels//ratio, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels//ratio),
                                    nn.ReLU())

        self.conv3_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels//ratio, out_channels, 1))
        self.conv3_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels//ratio, out_channels, 1))
        self.conv4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels//ratio, out_channels, 1))
    
    def forward(self, x):
        feat1 = self.conv1_1(x)
        pa_feat = self.pa(feat1)
        pa_conv = self.conv2_1(pa_feat)
        pa_out = self.conv3_1(pa_conv)

        feat2 = self.conv1_2(x)
        ca_feat = self.ca(feat2)
        ca_conv = self.conv2_2(ca_feat) 
        ca_out = self.conv3_2(ca_conv)

        feat_sum = pa_conv + ca_conv

        feat_sum_out = self.conv4(feat_sum)

        return pa_out, ca_out, feat_sum_out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7).cuda()
    danet=DAM(512,10).cuda()
    print(danet(input)[0].shape,danet(input)[1].shape,danet(input)[2].shape)
    print('okk!')