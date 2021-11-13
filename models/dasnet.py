# DASNet by CASIA IVA
# jliu@nlpr.ia.ac.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from danet import DAM
except:
    from models.danet import DAM
try:
    import resnet
except:
    from models import resnet
try:
    from models import hrnet
except:
    import hrnet

class ResNetBase(nn.Module):
    def __init__(self, in_planes, backbone):
        super(ResNetBase, self).__init__()
    
        if backbone == 'resnet50':
            self.basenet = resnet.ResNet50(in_planes, 2048)
        if backbone == 'resnet101':
            self.basenet = resnet.ResNet101(in_planes, 2048)
        if backbone == 'resnet152':
            self.basenet = resnet.ResNet152(in_planes, 2048)

    def base_forward(self, x):
        x = self.basenet.conv1(x)
        x = self.basenet.bn1(x)
        x = self.basenet.relu(x)
        x = self.basenet.maxpool(x)
        c1 = self.basenet.layer1(x) # 2048
        c2 = self.basenet.layer2(c1) # 2048
        c3 = self.basenet.layer3(c2) # 2048
        c4 = self.basenet.layer4(c3) # 2048
        return c1, c2, c3, c4

class DANet_ResNet(ResNetBase):
    def __init__(self, in_planes, out_planes, backbone, **kwargs):
        super(DANet_ResNet, self).__init__(in_planes, backbone, **kwargs)
        self.head = DAM(2048, out_planes)

    def forward(self, x):
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)

        return x[0],x[1],x[2]

class DANet_HRNet(nn.Module):
    def __init__(self, in_planes, out_planes, hrnet_config, **kwargs):
        super(DANet_HRNet, self).__init__()
        self.head = DAM(512, out_planes)
        self.net = hrnet.HighResolutionNet(hrnet_config, in_planes, 512)

    def forward(self, x):
        out = self.net(x)

        x = self.head(out)
        x = list(x)

        return x[0],x[1],x[2]

class SiameseNet(nn.Module):

    def __init__(self,in_planes, out_planes,base='resnet',norm_flag = 'l2'):
        super(SiameseNet, self).__init__()
        if norm_flag == 'l2':
            self.norm = F.normalize
        elif norm_flag == 'exp':
            self.norm = nn.Softmax2d()
        if base == 'resnet':
            self.Base = DANet_ResNet(in_planes, out_planes, backbone='resnet50')
        elif base == 'hrnet':
            self.Base = DANet_HRNet(in_planes, out_planes, hrnet_config='hrnet48')

    def forward(self,t0,t1):
        out_t0_conv5,out_t0_fc7,out_t0_embedding = self.Base(t0)
        out_t1_conv5,out_t1_fc7,out_t1_embedding = self.Base(t1)
        out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5,2,dim=1),self.norm(out_t1_conv5,2,dim=1)
        out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7,2,dim=1),self.norm(out_t1_fc7,2,dim=1)
        out_t0_embedding_norm,out_t1_embedding_norm = self.norm(out_t0_embedding,2,dim=1),self.norm(out_t1_embedding,2,dim=1)
        return [out_t0_conv5_norm,out_t1_conv5_norm],[out_t0_fc7_norm,out_t1_fc7_norm],[out_t0_embedding_norm,out_t1_embedding_norm]

if __name__ == '__main__':
    input1=torch.randn(4,4,256,256).cuda()
    input2=torch.randn(4,4,256,256).cuda()
    siamesenet= SiameseNet(4,12,base='hrnet').cuda()
    out = siamesenet(input1,input2)
    print(out[0][0].shape,out[0][1].shape)
    print(out[1][0].shape,out[1][1].shape)
    print(out[2][0].shape,out[2][1].shape)
    print('okk!')