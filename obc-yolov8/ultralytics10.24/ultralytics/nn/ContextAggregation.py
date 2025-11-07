#详细改进流程和操作，请关注B站博主：AI学术叫叫兽  持续更新哦


import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn import init as torch_init
#详细改进流程和操作，请关注B站博主：AI学术叫叫兽 

from mmcv.cnn import ConvModule
#详细改进流程和操作，请关注B站博主：AI学术叫叫兽 
 
class ContextAggregation(nn.Module):
#详细改进流程和操作，请关注B站博主：AI学术叫叫兽 
 
    def __init__(self, in_channels, reduction=1, conv_cfg=None):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)
 
        conv_params = dict(kernel_size=1, conv_cfg=conv_cfg, act_cfg=None)
 
        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)
 
        self.init_weights()
 
    def init_weights(self):
        # Replace deprecated mmcv initializers with torch equivalents
        for m in (self.a, self.k, self.v):
            if hasattr(m, 'conv') and hasattr(m.conv, 'weight') and m.conv.weight is not None:
                torch_init.xavier_uniform_(m.conv.weight)
            if hasattr(m, 'conv') and hasattr(m.conv, 'bias') and m.conv.bias is not None:
                torch_init.constant_(m.conv.bias, 0.0)
        if hasattr(self.m, 'conv') and hasattr(self.m.conv, 'weight') and self.m.conv.weight is not None:
            torch_init.constant_(self.m.conv.weight, 0.0)
        if hasattr(self.m, 'conv') and hasattr(self.m.conv, 'bias') and self.m.conv.bias is not None:
            torch_init.constant_(self.m.conv.bias, 0.0)
 
    def forward(self, x):
        n, c = x.size(0), self.inter_channels
 
        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()
 
        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)
 
        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)
 
        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a
 
        return x + y
#详细改进流程和操作，请关注B站博主：AI学术叫叫兽 
#pip install mmcv 