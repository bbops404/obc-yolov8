# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Convolution modules
"""

import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

__all__ = ('Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv')

# Global tracking for quantized vs FP32 fallback operations
_QUANTIZATION_STATS = defaultdict(int)
_ENABLE_QUANT_STATS = os.getenv('ENABLE_QUANT_STATS', '0') == '1'


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def _is_quantized_conv2d(module):
    """
    Check if a module is a quantized Conv2d.
    
    Args:
        module: Module to check
        
    Returns:
        bool: True if module is a quantized Conv2d
    """
    # Check for _packed_params first (definitive sign of quantized Conv2d)
    if hasattr(module, '_packed_params'):
        return True
    
    # Check if it's an instance of QuantizedConv2d (if available)
    try:
        from torch.ao.nn.quantized.modules.conv import Conv2d as QuantizedConv2d
        if isinstance(module, QuantizedConv2d):
            return True
    except ImportError:
        pass
    
    # Check if it's from quantized modules namespace
    module_type = type(module).__module__
    if module_type and ('quantized' in module_type or 'torch.ao.nn.quantized' in module_type):
        return True
    
    # If it's a Conv2d but not quantized, return False
    # Note: We check this last because quantized Conv2d might not be instance of nn.Conv2d
    if isinstance(module, nn.Conv2d):
        return False
    
    return False


def _safe_conv2d_call(conv_module, x, module_name=None):
    """
    Safely call a Conv2d module, handling both regular and quantized Conv2d.
    
    Args:
        conv_module: Conv2d module (regular or quantized)
        x: Input tensor (FP32 or quantized)
        module_name: Optional name for tracking/statistics
        
    Returns:
        Output tensor (FP32)
    """
    # Try normal call first (for regular Conv2d)
    # If it fails with AttributeError about _backward_hooks, it's likely quantized
    try:
        result = conv_module(x)
        if _ENABLE_QUANT_STATS:
            _QUANTIZATION_STATS['regular_conv2d'] += 1
        return result
    except AttributeError as e:
        msg = str(e)
        if (
            '_backward_hooks' in msg
            or '_forward_hooks' in msg
            or '_backward_pre_hooks' in msg
            or '_forward_pre_hooks' in msg
        ):
            # This is a quantized Conv2d - need to handle input quantization
            return _call_quantized_conv2d(conv_module, x, module_name)
        # Different AttributeError - re-raise
        raise
    except (NotImplementedError, RuntimeError) as e:
        # If we get NotImplementedError about quantized ops, try to quantize input first
        error_str = str(e)
        if 'quantized::' in error_str or 'QuantizedCPU' in error_str:
            return _call_quantized_conv2d(conv_module, x, module_name)
        else:
            raise


def _call_quantized_conv2d(conv_module, x, module_name=None):
    """
    Call a quantized Conv2d module with proper input quantization.
    
    Args:
        conv_module: Quantized Conv2d module
        x: Input tensor (FP32 or quantized)
        module_name: Optional name for tracking/statistics
        
    Returns:
        Output tensor (FP32, dequantized)
    """
    # Ensure input is FP32 - dequantize if needed
    if hasattr(x, 'q_scale') and hasattr(x, 'q_zero_point'):
        # Input is already quantized, dequantize it first
        x = x.dequantize()
    elif x.dtype != torch.float32:
        # If input is not FP32 and not quantized, convert it
        if x.dtype in (torch.qint8, torch.quint8, torch.qint32):
            # It's a quantized dtype but not a QuantizedTensor - try to convert
            x = x.float()
        else:
            x = x.float()
    
    # Input is now FP32 - need to quantize it before passing to quantized Conv2d
    # For activations, we use quint8 (unsigned int8, range 0-255)
    # Calculate scale and zero_point from input tensor
    x_min = x.min().item()
    x_max = x.max().item()
    
    # Handle edge case where all values are the same
    if x_max == x_min:
        scale = 1.0
        zero_point = 128  # Middle of quint8 range
    else:
        # Calculate scale and zero_point for quint8 (affine quantization)
        # Range: [0, 255]
        # Scale: (max - min) / 255
        scale = (x_max - x_min) / 255.0
        # Zero point: maps 0.0 to a value in [0, 255]
        # zero_point = round(-min / scale)
        zero_point = int(round(-x_min / scale))
        # Clamp zero_point to valid range
        zero_point = max(0, min(255, zero_point))
    
    # Ensure scale is not zero
    if scale == 0.0:
        scale = 1.0
    
    # Quantize input tensor
    try:
        x_quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    except Exception as quantize_error:
        # If quantization fails, try with default values
        # Use a reasonable default scale based on input range
        scale = max(abs(x_min), abs(x_max)) / 127.0 if max(abs(x_min), abs(x_max)) > 0 else 1.0
        zero_point = 128
        try:
            x_quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        except Exception:
            # Last resort: use very simple quantization
            scale = 0.01
            zero_point = 128
            x_quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    
    # Call quantized Conv2d forward with quantized input
    try:
        output = conv_module.forward(x_quantized)
        # Successfully used quantized operation
        if _ENABLE_QUANT_STATS:
            _QUANTIZATION_STATS['quantized_conv2d_success'] += 1
            if module_name:
                _QUANTIZATION_STATS[f'quantized_success_{module_name}'] += 1
        # Dequantize output to FP32 before returning (activations need FP32)
        if hasattr(output, 'q_scale') and hasattr(output, 'q_zero_point'):
            return output.dequantize()
        return output
    except (NotImplementedError, RuntimeError) as forward_error:
        # Check if it's a backend dispatch error
        error_msg = str(forward_error)
        if ('quantized::conv2d' in error_msg or 'quantized::conv' in error_msg) and ('CPU' in error_msg or 'backend' in error_msg.lower()):
            # Backend dispatch error - fallback to FP32 convolution
            if _ENABLE_QUANT_STATS:
                _QUANTIZATION_STATS['quantized_conv2d_fallback'] += 1
                if module_name:
                    _QUANTIZATION_STATS[f'quantized_fallback_{module_name}'] += 1
            
            input_fp32 = x_quantized.dequantize() if hasattr(x_quantized, 'q_scale') else x_quantized
            
            # Extract weights and bias from quantized Conv2d
            if hasattr(conv_module, '_packed_params') and conv_module._packed_params is not None:
                weight, bias = conv_module._packed_params
                weight_fp32 = weight.dequantize()
                bias_fp32 = bias.dequantize() if bias is not None else None
                
                # Perform FP32 convolution
                output_fp32 = torch.nn.functional.conv2d(
                    input_fp32, weight_fp32, bias_fp32,
                    stride=conv_module.stride, padding=conv_module.padding,
                    dilation=conv_module.dilation, groups=conv_module.groups
                )
                return output_fp32
            else:
                # If packed_params not available, try to get weight directly
                if hasattr(conv_module, 'weight'):
                    weight_fp32 = conv_module.weight.dequantize() if hasattr(conv_module.weight, 'q_scale') else conv_module.weight.float()
                    bias_fp32 = None
                    if hasattr(conv_module, 'bias') and conv_module.bias is not None:
                        bias_fp32 = conv_module.bias.dequantize() if hasattr(conv_module.bias, 'q_scale') else conv_module.bias.float()
                    
                    output_fp32 = torch.nn.functional.conv2d(
                        input_fp32, weight_fp32, bias_fp32,
                        stride=conv_module.stride, padding=conv_module.padding,
                        dilation=conv_module.dilation, groups=conv_module.groups
                    )
                    return output_fp32
        
        # If it's a different error, check for other known issues
        if 'Quantize only works on Float Tensor' in error_msg:
            # The quantized Conv2d is receiving a non-float tensor internally
            # This suggests the module structure might be incorrect
            raise RuntimeError(
                f"Quantized Conv2d received invalid input type. "
                f"This may indicate the model wasn't properly converted to INT8. "
                f"Original error: {forward_error}"
            )
        # Re-raise other errors
        raise RuntimeError(f"Failed to call quantized Conv2d forward: {forward_error}")
    
    # Output is quantized, dequantize it to FP32
    if hasattr(output, 'q_scale') and hasattr(output, 'q_zero_point'):
        return output.dequantize()
    
    # If output is not quantized (shouldn't happen), return as-is
    return output


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # Use safe call to handle both regular and quantized Conv2d
        conv_out = _safe_conv2d_call(self.conv, x)
        return self.act(self.bn(conv_out))

    def forward_fuse(self, x):
        """Perform fused convolution (BN already fused into conv) and activation."""
        # Use safe call to handle both regular and quantized Conv2d
        # Quantized Conv2d doesn't have _backward_hooks, so we need to bypass hook checking
        conv_out = _safe_conv2d_call(self.conv, x)
        return self.act(conv_out)


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # Use safe call to handle both regular and quantized Conv2d
        conv_out = _safe_conv2d_call(self.conv, x)
        cv2_out = _safe_conv2d_call(self.cv2, x)
        return self.act(self.bn(conv_out + cv2_out))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        # Use safe call to handle both regular and quantized Conv2d
        conv_out = _safe_conv2d_call(self.conv, x)
        return self.act(self.bn(conv_out))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution with args(ch_in, ch_out, kernel).
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status. This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
