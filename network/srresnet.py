import torch
import torch.nn as nn
import math
from PIL import Image
from network.PosEmbedding import PosEmbedding2D
from network.GANet import GANet_Conv
from network.mynn import initialize_weights, Norm2d, Norm1d
import numpy as np
import torch.nn.functional as F


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

class _NetG(nn.Module):
    def __init__(self, num_channels=2, use_pos = True, r_factor=2, pos_rfactor=2, pooling='mean'):
        super(_NetG, self).__init__()

        self.use_pos = use_pos
        self.num_channels = num_channels
        self.r_factor = r_factor
        self.pos_rfactor = pos_rfactor
        self.pooling = pooling
        
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.residual = self.make_layer(_Residual_Block, 64)

        if self.use_pos:
            self.pos_h = torch.arange(0, 213).unsqueeze(0).unsqueeze(2).expand(-1,-1,321)//2
            self.pos_w = torch.arange(0, 321).unsqueeze(0).unsqueeze(1).expand(-1,213,-1)//2
            
            self.pos_h = self.pos_h[0].byte().numpy()
            self.pos_w = self.pos_w[0].byte().numpy()
            # pos index to image
            self.pos_h = Image.fromarray(self.pos_h, mode="L")
            self.pos_w = Image.fromarray(self.pos_w, mode="L")
        
        ganet_in_channels = 64
        self.ganet0 = GANet_Conv(ganet_in_channels, 64, r_factor=self.r_factor, pos_rfactor=self.pos_rfactor, pooling=self.pooling)
        initialize_weights(self.ganet0)
        self.ganet1 = GANet_Conv(ganet_in_channels, 64, r_factor=self.r_factor, pos_rfactor=self.pos_rfactor, pooling=self.pooling)
        initialize_weights(self.ganet1)


        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True) 

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=9, stride=1, padding=4, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_pos:
            pos_h = torch.from_numpy(np.array(self.pos_h, dtype=np.uint8))# // self.pos_rfactor
            pos_w = torch.from_numpy(np.array(self.pos_w, dtype=np.uint8))# // self.pos_rfactor
            pos = (pos_h, pos_w)
        else:
            pos = None
        
        # Main branch
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        ganet_input = residual

        if self.use_pos:
            out = self.ganet0(ganet_input, out, pos)
            represent = out
            out = self.bn_mid(self.conv_mid(out))   
            out = self.ganet1(represent, out, pos)
        else:
            out = self.bn_mid(self.conv_mid(out))                
        
        
        out = torch.add(out,residual)
        out = self.conv_output(out)
        
        return out