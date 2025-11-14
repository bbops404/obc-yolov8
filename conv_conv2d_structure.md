# Conv Module and Conv2d Structure

## Overview

This document shows what the Conv module and its inner Conv2d layer look like before and after quantization.

---

## Conv Module Structure

### Before Quantization (FP32)

```python
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()  # or other activation
```

**Structure**:
```
Conv (wrapper module)
├── conv: torch.nn.Conv2d
│   ├── weight: torch.Tensor (FP32, shape=[out_channels, in_channels, kernel, kernel])
│   └── bias: None (bias=False, BN handles it)
├── bn: torch.nn.BatchNorm2d
│   ├── weight: torch.Tensor (FP32, shape=[out_channels])
│   ├── bias: torch.Tensor (FP32, shape=[out_channels])
│   ├── running_mean: torch.Tensor
│   └── running_var: torch.Tensor
└── act: nn.SiLU (or other activation)
```

**Forward Pass**:
```python
def forward(self, x):
    conv_out = self.conv(x)      # FP32 convolution
    bn_out = self.bn(conv_out)   # FP32 batch norm
    return self.act(bn_out)      # FP32 activation
```

---

### After Quantization (INT8)

**Structure**:
```
Conv (wrapper module)
├── conv: torch.ao.nn.quantized.modules.conv.Conv2d
│   ├── scale: torch.Tensor (float, per-channel for weights)
│   ├── zero_point: torch.Tensor (int, per-channel for weights)
│   └── _packed_params: PackedParams
│       ├── weight: torch.Tensor (qint8, quantized INT8 weights)
│       └── bias: torch.Tensor (qint32, quantized INT8 bias)
├── bn: (FUSED - doesn't exist as separate layer)
└── act: (FUSED - doesn't exist as separate layer)
```

**Key Differences**:
- `self.conv` is now `torch.ao.nn.quantized.modules.conv.Conv2d` (not `torch.nn.Conv2d`)
- BatchNorm and Activation are **fused** into the Conv2d operation
- Weights are stored in `_packed_params` as INT8 (qint8)
- Has `scale` and `zero_point` for quantization/dequantization

**Forward Pass**:
```python
def forward(self, x):
    # Input x is FP32
    # Conv2d internally:
    #   1. Quantizes input to INT8 (using activation scale/zero_point)
    #   2. Performs INT8 convolution (fast!)
    #   3. Dequantizes output back to FP32
    conv_out = self.conv(x)  # INT8 computation, FP32 output
    # BN and activation are fused, so no separate calls needed
    return conv_out
```

---

## Conv2d Layer Details

### FP32 Conv2d (Before Quantization)

```python
# Type: torch.nn.Conv2d
# Module: torch.nn.modules.conv

conv2d = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

# Attributes:
conv2d.weight          # torch.Tensor, shape=[128, 64, 3, 3], dtype=torch.float32
conv2d.bias            # None (bias=False)
conv2d.in_channels    # 64
conv2d.out_channels   # 128
conv2d.kernel_size   # (3, 3)
conv2d.stride         # (1, 1)
conv2d.padding         # (1, 1)

# No quantization parameters:
# - No scale
# - No zero_point
# - No _packed_params
```

**Memory**: 
- Weight size: `128 * 64 * 3 * 3 * 4 bytes = 294,912 bytes` (FP32)

---

### INT8 Conv2d (After Quantization)

```python
# Type: torch.ao.nn.quantized.modules.conv.Conv2d
# Module: torch.ao.nn.quantized.modules.conv

# From your model (model.0.conv):
conv2d = torch.ao.nn.quantized.modules.conv.Conv2d(...)

# Attributes:
conv2d.scale           # torch.Tensor, shape=[128], dtype=torch.float32
                       # Example: scale=0.59553689 (per-channel)
conv2d.zero_point      # torch.Tensor, shape=[128], dtype=torch.int32
                       # Example: zero_point=68 (per-channel)
conv2d._packed_params  # PackedParams object
    .weight           # torch.Tensor, shape=[128, 64, 3, 3], dtype=torch.qint8
    .bias             # torch.Tensor, shape=[128], dtype=torch.qint32

# Standard attributes (still available):
conv2d.in_channels     # 64
conv2d.out_channels   # 128
conv2d.kernel_size   # (3, 3)
conv2d.stride         # (1, 1)
conv2d.padding        # (1, 1)
```

**Memory**:
- Weight size: `128 * 64 * 3 * 3 * 1 byte = 73,728 bytes` (INT8)
- Scale size: `128 * 4 bytes = 512 bytes` (FP32)
- Zero point size: `128 * 4 bytes = 512 bytes` (INT32)
- **Total: ~74,752 bytes** (vs 294,912 bytes FP32)
- **Memory reduction: ~75%**

---

## Example from Your Model

### model.0.conv (First Layer)

**Before Quantization (FP32)**:
```python
model.0.conv
├── Type: torch.nn.Conv2d
├── Weight: torch.Tensor, shape=[64, 3, 6, 6], dtype=float32
├── Size: ~27,648 bytes
└── Computation: FP32 (slow)
```

**After Quantization (INT8)**:
```python
model.0.conv
├── Type: torch.ao.nn.quantized.modules.conv.Conv2d
├── Scale: 0.59553689 (per-channel, 64 values)
├── Zero Point: 68 (per-channel, 64 values)
├── _packed_params:
│   ├── Weight: torch.Tensor, shape=[64, 3, 6, 6], dtype=qint8
│   └── Bias: torch.Tensor, shape=[64], dtype=qint32
├── Size: ~6,912 bytes (75% reduction!)
└── Computation: INT8 (fast!)
```

---

## Quantization Process

### Step 1: QAT Preparation
```python
# Before: torch.nn.Conv2d
conv2d = nn.Conv2d(64, 128, 3)

# After prepare_for_qat():
# - Wrapped with FakeQuantize
# - Still torch.nn.Conv2d, but has:
conv2d.weight_fake_quant  # FakeQuantize module
conv2d.activation_post_process  # FakeQuantize module
```

### Step 2: QAT Training
```python
# During training:
# - Weights are FP32 but quantization noise is applied
# - Observers collect min/max statistics
# - Model learns to be robust to quantization
```

### Step 3: INT8 Conversion
```python
# After convert_to_quantized():
# - torch.nn.Conv2d → torch.ao.nn.quantized.modules.conv.Conv2d
# - FP32 weights → INT8 weights (in _packed_params)
# - Statistics → scale and zero_point
# - FakeQuantize removed
```

---

## Visual Comparison

### FP32 Conv Module
```
┌─────────────────────────────────┐
│         Conv Module             │
│  ┌───────────────────────────┐ │
│  │   nn.Conv2d (FP32)        │ │
│  │   Weight: [128,64,3,3]    │ │
│  │   Dtype: float32            │ │
│  │   Size: 294KB              │ │
│  └───────────────────────────┘ │
│  ┌───────────────────────────┐ │
│  │   BatchNorm2d (FP32)      │ │
│  └───────────────────────────┘ │
│  ┌───────────────────────────┐ │
│  │   SiLU Activation (FP32) │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
```

### INT8 Conv Module
```
┌─────────────────────────────────┐
│         Conv Module             │
│  ┌───────────────────────────┐ │
│  │ QuantizedConv2d (INT8)   │ │
│  │ ┌─────────────────────┐  │ │
│  │ │ _packed_params:     │  │ │
│  │ │  Weight: qint8      │  │ │
│  │ │  Bias: qint32       │  │ │
│  │ │  Size: 74KB         │  │ │
│  │ └─────────────────────┘  │ │
│  │ Scale: [0.5955, ...]     │ │
│  │ Zero Point: [68, ...]    │ │
│  │ (BN + Act fused)          │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
```

---

## Key Takeaways

1. **Conv Module**: Wrapper that contains Conv2d, BatchNorm, and Activation
2. **Before Quantization**: 
   - `self.conv` is `torch.nn.Conv2d` (FP32)
   - Separate BN and activation layers
   - Large memory footprint
3. **After Quantization**:
   - `self.conv` is `torch.ao.nn.quantized.modules.conv.Conv2d` (INT8)
   - BN and activation are **fused** into Conv2d
   - Weights stored in `_packed_params` as INT8
   - Has `scale` and `zero_point` for quantization
   - **75% memory reduction**
   - **Faster inference** (INT8 operations)

4. **Fusion**: BatchNorm and Activation are fused into Conv2d during quantization preparation, so they don't exist as separate layers in the INT8 model.

