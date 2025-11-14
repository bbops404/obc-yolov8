"""
Quantization-Aware Training (QAT) modules for YOLOv8-CA
Implements QAT versions of custom modules with proper quantization boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub

from ultralytics.nn.modules import Conv
from ultralytics.nn.BoTNet import BoTNet, MHSA, BottleneckTransformer
from ultralytics.nn.CA_Attention import CoordAtt
from ultralytics.nn.ODConv import ODConv


class QATMHSA(nn.Module):
    """
    Quantization-Aware Multi-Head Self-Attention
    
    Key features:
    - Q, K, V Conv2d projections are auto-quantized by FX
    - MatMul operations are quantized
    - Softmax kept in FP32 for numerical stability
    """
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(QATMHSA, self).__init__()
        
        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(
                torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                requires_grad=True
            )
            self.rel_w_weight = nn.Parameter(
                torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                requires_grad=True
            )
        
        # Critical: Dequantize before softmax to keep it in FP32
        self.dequant_pre_softmax = DeQuantStub()
        
        # Re-quantize after softmax for second matmul
        self.quant_post_softmax = QuantStub()
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        n_batch, C, width, height = x.size()
        
        # Q, K, V projections (Conv2d will be auto-quantized by FX)
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        
        # First matmul: Q @ K^T (will be quantized by FX)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        
        c1, c2, c3, c4 = content_content.size()
        
        if self.pos:
            # Positional encoding
            content_position = (self.rel_h_weight + self.rel_w_weight).view(
                1, self.heads, C // self.heads, -1
            ).permute(0, 1, 3, 2)
            
            content_position = torch.matmul(content_position, q)
            content_position = content_position if (
                content_content.shape == content_position.shape
            ) else content_position[:, :, :c3, ]
            
            energy = content_content + content_position
        else:
            energy = content_content
        
        # CRITICAL: Dequantize before softmax for numerical stability
        energy = self.dequant_pre_softmax(energy)
        
        # Softmax in FP32
        attention = self.softmax(energy)
        
        # Re-quantize after softmax
        attention = self.quant_post_softmax(attention)
        
        # Second matmul: Attention @ V (will be quantized by FX)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        
        return out


class QATBottleneckTransformer(nn.Module):
    """
    Quantization-Aware Transformer Bottleneck
    Uses QATMHSA for proper attention quantization
    """
    def __init__(self, c1, c2, stride=1, heads=4, mhsa=True, resolution=None, expansion=1):
        super(QATBottleneckTransformer, self).__init__()
        c_ = int(c2 * expansion)
        self.cv1 = Conv(c1, c_, 1, 1)
        
        if not mhsa:
            self.cv2 = Conv(c_, c2, 3, 1)
        else:
            self.cv2 = nn.ModuleList()
            # Use QAT-aware MHSA
            self.cv2.append(QATMHSA(c2, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2d(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)
        
        self.shortcut = c1 == c2
        if stride != 1 or c1 != expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, expansion * c2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion * c2)
            )
        
        # Linear layer will be auto-quantized by FX
        self.fc1 = nn.Linear(c2, c2)
    
    def forward(self, x):
        out = x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))
        return out


class QATBoTNet(nn.Module):
    """
    Quantization-Aware BoTNet module
    Wraps the BoTNet with quantization boundaries
    """
    def __init__(self, c1, c2, n=1, e=0.5, e2=1, w=20, h=20):
        super(QATBoTNet, self).__init__()
        
        # Input quantization stub
        self.quant = QuantStub()
        
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        
        # Use QAT-aware BottleneckTransformer
        self.m = nn.Sequential(
            *[QATBottleneckTransformer(c_, c_, stride=1, heads=4, mhsa=True, 
                                       resolution=(w, h), expansion=e2) for _ in range(n)]
        )
        
        # Output dequantization stub
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        out = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        out = self.dequant(out)
        return out


class QATCoordAtt(nn.Module):
    """
    Quantization-Aware Coordinate Attention module
    Wraps CoordAtt with quantization boundaries
    """
    def __init__(self, inp, reduction=32):
        super(QATCoordAtt, self).__init__()
        
        # Input quantization stub
        self.quant = QuantStub()
        
        # Original CoordAtt module (Conv2d will be auto-quantized)
        self.coordatt = CoordAtt(inp, reduction)
        
        # Output dequantization stub
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.coordatt(x)
        x = self.dequant(x)
        return x


class FP32ODConv(nn.Module):
    """
    FP32 wrapper for ODConv
    Forces ODConv to remain in FP32 (unquantized)
    
    ODConv uses dynamic kernel aggregation which is incompatible with quantization
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, 
                 norm_layer=nn.BatchNorm2d, reduction=0.0625, kernel_num=1):
        super(FP32ODConv, self).__init__()
        
        # Original ODConv module
        self.odconv = ODConv(in_planes, out_planes, kernel_size, stride, groups, 
                            norm_layer, reduction, kernel_num)
        
        # Explicitly set qconfig to None to disable quantization
        self.qconfig = None
    
    def forward(self, x):
        # Ensure input is FP32 (in case coming from quantized layer)
        if x.dtype != torch.float32:
            x = x.float()
        
        out = self.odconv(x)
        return out


# Helper function to replace modules with QAT versions
def replace_with_qat_modules(model, qat_mode=True):
    """
    Replace standard modules with QAT-aware versions
    
    Args:
        model: The model to modify
        qat_mode: If True, use QAT versions; if False, use FP32 wrappers where needed
    
    Returns:
        Modified model
    """
    if not qat_mode:
        return model
    
    for name, module in model.named_children():
        # Recursively replace in children
        if len(list(module.children())) > 0:
            replace_with_qat_modules(module, qat_mode)
        
        # Replace BoTNet with QATBoTNet
        if isinstance(module, BoTNet):
            # Get original parameters
            c1 = module.cv1.conv.in_channels
            c2 = module.cv3.conv.out_channels
            # Note: n, e, e2, w, h would need to be stored or inferred
            qat_module = QATBoTNet(c1, c2)
            # Copy YOLO-specific attributes (f, i, type) if they exist
            if hasattr(module, 'f'):
                qat_module.f = module.f
            if hasattr(module, 'i'):
                qat_module.i = module.i
            if hasattr(module, 'type'):
                qat_module.type = module.type
            setattr(model, name, qat_module)
        
        # Replace CoordAtt with QATCoordAtt
        elif isinstance(module, CoordAtt):
            inp = module.conv1.in_channels
            mip = module.conv1.out_channels
            # Infer reduction: mip = max(8, inp // reduction)
            # If mip >= 8 and mip < inp, then reduction = inp // mip
            # Otherwise, use default reduction of 32
            if mip >= 8 and mip < inp:
                reduction = inp // mip
            else:
                reduction = 32  # Default reduction
            qat_module = QATCoordAtt(inp, reduction)
            # Copy weights from original CoordAtt to QATCoordAtt's internal CoordAtt
            qat_module.coordatt.load_state_dict(module.state_dict())
            # Copy YOLO-specific attributes (f, i, type) if they exist
            if hasattr(module, 'f'):
                qat_module.f = module.f
            if hasattr(module, 'i'):
                qat_module.i = module.i
            if hasattr(module, 'type'):
                qat_module.type = module.type
            setattr(model, name, qat_module)
        
        # Replace ODConv with FP32ODConv
        elif isinstance(module, ODConv):
            # Wrap with FP32 version
            in_planes = module[0].in_planes if hasattr(module, '__getitem__') else None
            if in_planes:
                out_planes = module[0].out_planes
                qat_module = FP32ODConv(in_planes, out_planes)
                # Copy YOLO-specific attributes (f, i, type) if they exist
                if hasattr(module, 'f'):
                    qat_module.f = module.f
                if hasattr(module, 'i'):
                    qat_module.i = module.i
                if hasattr(module, 'type'):
                    qat_module.type = module.type
                setattr(model, name, qat_module)
    
    return model

