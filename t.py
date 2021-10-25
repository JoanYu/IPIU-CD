from torch import tensor

import numpy as np
import tifffile
import torch

from models.hrnet import HighResolutionNet as HRNet

img1 = torch.tensor(tifffile.imread('data/train/A/00403_0_0.tif').transpose(2,0,1)).unsqueeze(0)
img2 = torch.tensor(tifffile.imread('data/train/A/00403_0_1.tif').transpose(2,0,1)).unsqueeze(0)
img3 = torch.tensor(tifffile.imread('data/train/A/00403_1_0.tif').transpose(2,0,1)).unsqueeze(0)
img4 = torch.tensor(tifffile.imread('data/train/A/00403_1_1.tif').transpose(2,0,1)).unsqueeze(0)

tensor = torch.cat((img1,img2,img3,img4),dim = 0)
tensor = tensor.float()
tensor = tensor.cuda()
print(tensor.shape)

t = HRNet('hrnet64',6)
t = t.cuda()
y = t(tensor)
print(y.shape)