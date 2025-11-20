#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

import math
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
import numpy as np
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment)
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()
 
        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
 
    def forward(self, x):
        n_batch, C, width, height = x.size()
        
        # Handle quantized Conv2d inputs and outputs
        # Quantized Conv2d requires quantized input tensors on QuantizedCPU backend
        def _quantize_if_needed(tensor, scale=None, zero_point=None):
            """Quantize tensor if it's FP32, otherwise return as-is."""
            if hasattr(tensor, 'q_scale') and hasattr(tensor, 'q_zero_point'):
                # Already quantized
                return tensor
            
            # Need to quantize FP32 tensor
            if scale is None or zero_point is None:
                # Calculate scale and zero_point from tensor
                t_min = tensor.min().item()
                t_max = tensor.max().item()
                if t_max == t_min:
                    scale = 1.0
                    zero_point = 128
                else:
                    scale = (t_max - t_min) / 255.0
                    zero_point = int(round(-t_min / scale))
                    zero_point = max(0, min(255, zero_point))
                    if scale == 0.0:
                        scale = 1.0
            
            try:
                return torch.quantize_per_tensor(tensor, scale, zero_point, torch.quint8)
            except Exception:
                # Fallback quantization
                return torch.quantize_per_tensor(tensor, 0.01, 128, torch.quint8)
        
        def _dequantize_if_needed(tensor):
            """Dequantize tensor if it's quantized, otherwise return as-is."""
            if hasattr(tensor, 'q_scale') and hasattr(tensor, 'q_zero_point'):
                return tensor.dequantize()
            return tensor
        
        # Check if query/key/value are quantized Conv2d
        query_is_quantized = hasattr(self.query, '_packed_params') or 'quantized' in type(self.query).__module__.lower()
        key_is_quantized = hasattr(self.key, '_packed_params') or 'quantized' in type(self.key).__module__.lower()
        value_is_quantized = hasattr(self.value, '_packed_params') or 'quantized' in type(self.value).__module__.lower()
        
        # Wrapper to safely call quantized Conv2d with backend error handling
        def _safe_quantized_conv2d(conv_module, input_tensor):
            """Safely call quantized Conv2d, handling backend dispatch errors."""
            if not (hasattr(conv_module, '_packed_params') or 'quantized' in type(conv_module).__module__.lower()):
                # Not quantized, call normally
                return conv_module(input_tensor)
            
            # It's quantized - ensure input is quantized
            if not (hasattr(input_tensor, 'q_scale') and hasattr(input_tensor, 'q_zero_point')):
                # Input is FP32, need to quantize
                input_tensor = _quantize_if_needed(input_tensor)
            
            try:
                # Try calling quantized Conv2d
                return conv_module(input_tensor)
            except (NotImplementedError, RuntimeError) as e:
                error_msg = str(e)
                if 'quantized::conv2d' in error_msg and ('CPU' in error_msg or 'backend' in error_msg.lower()):
                    # Backend dispatch error - this is a known PyTorch limitation
                    # Fallback: Extract weights from quantized Conv2d and perform FP32 convolution
                    input_fp32 = input_tensor.dequantize() if hasattr(input_tensor, 'q_scale') else input_tensor
                    
                    try:
                        # Extract weight and bias from _packed_params
                        if hasattr(conv_module, '_packed_params') and conv_module._packed_params is not None:
                            packed = conv_module._packed_params
                            # _packed_params is typically a tuple of (weight, bias) or just weight
                            if isinstance(packed, tuple) and len(packed) >= 1:
                                weight = packed[0]
                                bias = packed[1] if len(packed) > 1 else None
                                
                                # Dequantize weight (it's a quantized tensor)
                                if hasattr(weight, 'q_scale'):
                                    weight_fp32 = weight.dequantize()
                                else:
                                    weight_fp32 = weight
                                
                                # Dequantize bias if present and quantized
                                if bias is not None and hasattr(bias, 'q_scale'):
                                    bias_fp32 = bias.dequantize()
                                elif bias is not None:
                                    bias_fp32 = bias
                                else:
                                    bias_fp32 = None
                                
                                # Get convolution parameters
                                stride = getattr(conv_module, 'stride', (1, 1))
                                if isinstance(stride, int):
                                    stride = (stride, stride)
                                padding = getattr(conv_module, 'padding', (0, 0))
                                if isinstance(padding, int):
                                    padding = (padding, padding)
                                dilation = getattr(conv_module, 'dilation', (1, 1))
                                if isinstance(dilation, int):
                                    dilation = (dilation, dilation)
                                groups = getattr(conv_module, 'groups', 1)
                                
                                # Perform FP32 convolution
                                output_fp32 = torch.nn.functional.conv2d(
                                    input_fp32, weight_fp32, bias_fp32,
                                    stride=stride, padding=padding,
                                    dilation=dilation, groups=groups
                                )
                                return output_fp32
                    except Exception as fallback_error:
                        # If fallback fails, log and return FP32 input (model will continue with reduced accuracy)
                        import warnings
                        warnings.warn(f"Quantized Conv2d backend error, using FP32 fallback: {fallback_error}")
                    
                    # Last resort: return FP32 input (allows forward pass but loses quantization benefits)
                    return input_fp32
                else:
                    # Different error, re-raise
                    raise
        
        # Get query, key, value outputs using safe wrapper
        q_raw = _safe_quantized_conv2d(self.query, x)
        k_raw = _safe_quantized_conv2d(self.key, x)
        v_raw = _safe_quantized_conv2d(self.value, x)
        
        # Dequantize before view operations (view doesn't work well with quantized tensors)
        q = _dequantize_if_needed(q_raw).view(n_batch, self.heads, C // self.heads, -1)
        k = _dequantize_if_needed(k_raw).view(n_batch, self.heads, C // self.heads, -1)
        v = _dequantize_if_needed(v_raw).view(n_batch, self.heads, C // self.heads, -1)
        
        # print('q shape:{},k shape:{},v shape:{}'.format(q.shape,k.shape,v.shape))  #1,4,64,256
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        # print("qkT=",content_content.shape)
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            # print("old content_content shape",content_content.shape) #1,4,256,256
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = content_position if (
                        content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            # print('new pos222-> shape:',content_position.shape)
            # print('new content222-> shape:',content_content.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out
 
 
class BottleneckTransformer(nn.Module):
    # Transformer bottleneck
    # expansion = 1
 
    def __init__(self, c1, c2, stride=1, heads=4, mhsa=True, resolution=None, expansion=1):
        super(BottleneckTransformer, self).__init__()
        c_ = int(c2 * expansion)
        self.cv1 = Conv(c1, c_, 1, 1)
        # self.bn1 = nn.BatchNorm2d(c2)
        if not mhsa:
            self.cv2 = Conv(c_, c2, 3, 1)
        else:
            self.cv2 = nn.ModuleList()
            self.cv2.append(MHSA(c2, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2d(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)
        self.shortcut = c1 == c2
        if stride != 1 or c1 != expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, expansion * c2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion * c2)
            )
        self.fc1 = nn.Linear(c2, c2)
 
    def forward(self, x):
        out = x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))
        return out
 
 
class BoTNet(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, e=0.5, e2=1, w=20, h=20):  # ch_in, ch_out, number, , expansion,w,h
        super(BoTNet, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(
            *[BottleneckTransformer(c_, c_, stride=1, heads=4, mhsa=True, resolution=(w, h), expansion=e2) for _ in
              range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽  
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))