# Module Quantization Strategy

This document explains how each module type in YOLOv8-CA is quantized during QAT (Quantization-Aware Training) and converted to INT8.

## Overview

The quantization process follows these steps:
1. **QAT Preparation**: Insert FakeQuantize modules to simulate quantization during training
2. **QAT Training**: Train with quantization simulation to learn robust quantized weights
3. **INT8 Conversion**: Replace FakeQuantize with real quantized operations (torch.ao.nn.quantized.modules.conv.Conv2d)

---

## 1. Conv Module

### Structure
```python
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        self.conv = nn.Conv2d(...)  # The actual Conv2d layer
        self.bn = nn.BatchNorm2d(c2)  # BatchNorm
        self.act = nn.SiLU()  # Activation
```

### Quantization Strategy

**During QAT:**
- **Conv+BN+Activation Fusion**: Before QAT, Conv+BN+Activation are fused into a single operation
- **Inner Conv2d Quantization**: The `self.conv` (Conv2d) layer receives a qconfig and gets wrapped with FakeQuantize
  - Weight quantization: Per-channel INT8 (qint8)
  - Activation quantization: Per-tensor INT8 (quint8)
- **BatchNorm**: Fused into Conv2d, so it doesn't exist as a separate layer after fusion
- **Activation**: Fused into Conv2d

**After INT8 Conversion:**
- `self.conv` becomes `torch.ao.nn.quantized.modules.conv.Conv2d`
- Has `_packed_params` containing quantized weights and biases
- Has `scale` and `zero_point` for quantization parameters
- BatchNorm and activation are part of the fused quantized operation

**Example from your model:**
- `model.0.conv`: scale=0.59553689, zero_point=68
- `model.2.cv1.conv`: scale=0.09886258, zero_point=70
- All 60 Conv wrapper layers have their inner Conv2d quantized

---

## 2. C2f Module

### Structure
```python
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Input projection
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Output projection
        self.m = nn.ModuleList([Bottleneck(...) for _ in range(n)])  # Bottleneck blocks
```

### Quantization Strategy

**During QAT:**
- **No special QAT wrapper**: C2f uses standard Conv modules internally
- **cv1 and cv2**: These are Conv modules, so their inner Conv2d layers get quantized
- **Bottleneck blocks (self.m)**: Each Bottleneck contains Conv layers, which get quantized
- **All Conv2d inside**: All Conv2d layers inside C2f receive qconfig and are quantized

**After INT8 Conversion:**
- `cv1.conv` and `cv2.conv` become quantized Conv2d
- All Conv2d inside Bottleneck blocks become quantized
- The module structure remains the same, only inner Conv2d layers are quantized

**Example from your model:**
- `model.2.cv1.conv`: scale=0.09886258, zero_point=70 (C2f input)
- `model.2.cv2.conv`: scale=0.08055609, zero_point=57 (C2f output)
- `model.2.m.0.cv1.conv`: scale=0.08670790, zero_point=66 (Bottleneck)
- `model.2.m.0.cv2.conv`: scale=0.05691754, zero_point=62 (Bottleneck)

**What's NOT quantized:**
- Concatenation operations (torch.cat) - these are just tensor operations
- Chunk/split operations - these are just tensor operations
- Residual addition (x + ...) - element-wise operations remain FP32

---

## 2.1. Bottleneck Module (inside C2f)

### Structure
```python
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        self.cv1 = Conv(c1, c_, k[0], 1)  # First Conv (usually 3x3)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # Second Conv (usually 3x3)
        self.add = shortcut and c1 == c2  # Residual connection flag
```

### Quantization Strategy

**During QAT:**
- **No special QAT wrapper**: Bottleneck uses standard Conv modules internally
- **cv1 and cv2**: These are Conv modules, so their inner Conv2d layers get quantized
- **Both Conv2d quantized**: Both `cv1.conv` and `cv2.conv` receive qconfig and are quantized
- **Residual connection**: The addition operation (if `self.add=True`) remains FP32

**After INT8 Conversion:**
- `cv1.conv` becomes `torch.ao.nn.quantized.modules.conv.Conv2d`
- `cv2.conv` becomes `torch.ao.nn.quantized.modules.conv.Conv2d`
- Residual addition remains FP32 (element-wise operation)

**Example from your model (all Bottleneck Conv2d layers are quantized):**

**C2f model.2 (1 Bottleneck):**
- `model.2.m.0.cv1.conv`: scale=0.08670790, zero_point=66 ✓ QUANTIZED
- `model.2.m.0.cv2.conv`: scale=0.05691754, zero_point=62 ✓ QUANTIZED

**C2f model.4 (2 Bottlenecks):**
- `model.4.m.0.cv1.conv`: scale=0.06151308, zero_point=60 ✓ QUANTIZED
- `model.4.m.0.cv2.conv`: scale=0.05571702, zero_point=65 ✓ QUANTIZED
- `model.4.m.1.cv1.conv`: scale=0.06862725, zero_point=58 ✓ QUANTIZED
- `model.4.m.1.cv2.conv`: scale=0.07145538, zero_point=80 ✓ QUANTIZED

**C2f model.6 (2 Bottlenecks):**
- `model.6.m.0.cv1.conv`: scale=0.06474771, zero_point=74 ✓ QUANTIZED
- `model.6.m.0.cv2.conv`: scale=0.06323306, zero_point=61 ✓ QUANTIZED
- `model.6.m.1.cv1.conv`: scale=0.06075667, zero_point=68 ✓ QUANTIZED
- `model.6.m.1.cv2.conv`: scale=0.05893991, zero_point=68 ✓ QUANTIZED

**C2f model.8 (1 Bottleneck):**
- `model.8.m.0.cv1.conv`: scale=0.06254458, zero_point=67 ✓ QUANTIZED
- `model.8.m.0.cv2.conv`: scale=0.06900568, zero_point=68 ✓ QUANTIZED

**C2f model.13 (1 Bottleneck):**
- `model.13.m.0.cv1.conv`: scale=0.05282248, zero_point=54 ✓ QUANTIZED
- `model.13.m.0.cv2.conv`: scale=0.05060664, zero_point=51 ✓ QUANTIZED

**C2f model.16 (1 Bottleneck):**
- `model.16.m.0.cv1.conv`: scale=0.03236512, zero_point=53 ✓ QUANTIZED
- `model.16.m.0.cv2.conv`: scale=0.03980798, zero_point=57 ✓ QUANTIZED

**C2f model.19 (1 Bottleneck):**
- `model.19.m.0.cv1.conv`: scale=0.04345312, zero_point=53 ✓ QUANTIZED
- `model.19.m.0.cv2.conv`: scale=0.03904831, zero_point=65 ✓ QUANTIZED

**C2f model.23 (1 Bottleneck):**
- `model.23.m.0.cv1.conv`: scale=0.04393817, zero_point=52 ✓ QUANTIZED
- `model.23.m.0.cv2.conv`: scale=0.03650638, zero_point=62 ✓ QUANTIZED

**Total: 9 Bottleneck modules with 18 quantized Conv2d layers (cv1 + cv2 for each)**

**What's NOT quantized:**
- Residual addition (x + cv2(cv1(x))) - element-wise addition remains FP32
- The Bottleneck module itself is just a container - only inner Conv2d are quantized

---

## 3. SPPF Module

### Structure
```python
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        self.cv1 = Conv(c1, c_, 1, 1)  # Input projection
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # Output projection
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # MaxPool
```

### Quantization Strategy

**During QAT:**
- **No special QAT wrapper**: SPPF uses standard Conv modules internally
- **cv1 and cv2**: These are Conv modules, so their inner Conv2d layers get quantized
- **MaxPool2d**: Remains in FP32 (pooling operations are typically not quantized)

**After INT8 Conversion:**
- `cv1.conv` and `cv2.conv` become quantized Conv2d
- MaxPool2d remains FP32
- Concatenation operations remain FP32

**Example from your model:**
- SPPF layers would have quantized `cv1.conv` and `cv2.conv` if present

**What's NOT quantized:**
- MaxPool2d operations
- Concatenation operations

---

## 4. BoTNet Module

### Structure
```python
class BoTNet(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, e2=1, w=20, h=20):
        self.cv1 = Conv(c1, c_, 1, 1)  # Input projection
        self.cv2 = Conv(c1, c_, 1, 1)  # Parallel branch
        self.cv3 = Conv(2 * c_, c2, 1)  # Output projection
        self.m = nn.Sequential([BottleneckTransformer(...)])  # MHSA blocks
```

### Quantization Strategy

**During QAT:**
- **QATBoTNet wrapper**: BoTNet is replaced with QATBoTNet during QAT preparation
- **QuantStub/DeQuantStub**: Input is quantized, output is dequantized
- **cv1, cv2, cv3**: These Conv modules have their inner Conv2d quantized
- **BottleneckTransformer (MHSA)**: 
  - Query, Key, Value Conv2d projections are quantized
  - MatMul operations are quantized
  - Softmax is kept in FP32 for numerical stability (dequantized before, requantized after)

**After INT8 Conversion:**
- `cv1.conv`, `cv2.conv`, `cv3.conv` become quantized Conv2d
- MHSA Conv2d layers (query, key, value) become quantized
- Softmax remains FP32

**Example from your model:**
- BoTNet layers would have quantized Conv2d inside cv1, cv2, cv3
- MHSA query/key/value Conv2d would be quantized

**What's NOT quantized:**
- Softmax operations (kept in FP32 for numerical stability)
- Positional embedding parameters (if used)

---

## 5. CoordAtt Module

### Structure
```python
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)  # Reduction conv
        self.bn1 = nn.BatchNorm2d(mip)  # BatchNorm
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)  # Horizontal attention
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)  # Vertical attention
```

### Quantization Strategy

**During QAT:**
- **QATCoordAtt wrapper**: CoordAtt is replaced with QATCoordAtt during QAT preparation
- **QuantStub/DeQuantStub**: Input is quantized, output is dequantized
- **conv1, conv_h, conv_w**: These Conv2d layers receive qconfig and are quantized
- **bn1**: BatchNorm is NOT fused (standalone), so it remains FP32

**After INT8 Conversion:**
- `conv1`, `conv_h`, `conv_w` become `torch.ao.nn.quantized.modules.conv.Conv2d`
- `bn1` remains FP32 (BatchNorm2d)
- AdaptiveAvgPool2d operations remain FP32

**Example from your model:**
- `model.20.conv1`: scale=0.04295848, zero_point=72 ✓ QUANTIZED
- `model.20.conv_h`: scale=0.03451281, zero_point=65 ✓ QUANTIZED
- `model.20.conv_w`: scale=0.02506289, zero_point=61 ✓ QUANTIZED
- `model.20.bn1`: FP32 (NOT quantized)
- `model.24.conv1`: scale=0.01631050, zero_point=61 ✓ QUANTIZED
- `model.24.conv_h`: scale=0.04412067, zero_point=64 ✓ QUANTIZED
- `model.24.conv_w`: scale=0.03503848, zero_point=67 ✓ QUANTIZED
- `model.24.bn1`: FP32 (NOT quantized)

**What's NOT quantized:**
- BatchNorm2d (bn1) - kept in FP32
- AdaptiveAvgPool2d operations
- Sigmoid activations (element-wise, typically not quantized)
- Element-wise multiplication operations

---

## Summary Table

| Module Type | Quantized Components | FP32 Components | Special Notes |
|------------|---------------------|-----------------|---------------|
| **Conv** | `self.conv` (Conv2d) | None (after fusion) | Conv+BN+Activation fused before quantization |
| **C2f** | All `cv1.conv`, `cv2.conv`, and Conv2d inside Bottleneck | Concatenation, chunk operations | Uses Conv modules internally |
| **Bottleneck** (inside C2f) | `cv1.conv`, `cv2.conv` (both Conv2d) | Residual addition (x + ...) | Each Bottleneck has 2 quantized Conv2d layers |
| **SPPF** | `cv1.conv`, `cv2.conv` | MaxPool2d, concatenation | Pooling operations not quantized |
| **BoTNet** | `cv1.conv`, `cv2.conv`, `cv3.conv`, MHSA query/key/value Conv2d | Softmax, positional embeddings | Wrapped with QATBoTNet, softmax in FP32 |
| **CoordAtt** | `conv1`, `conv_h`, `conv_w` | `bn1` (BatchNorm2d), pooling, sigmoid | Wrapped with QATCoordAtt, BN not fused |

---

## Quantization Parameters

Each quantized Conv2d has:
- **Scale**: Floating-point value that maps quantized integers to real values
  - Formula: `real_value = (quantized_value - zero_point) * scale`
- **Zero Point**: Integer offset for quantization
  - Typically in range [0, 255] for quint8 activations
- **_packed_params**: Contains quantized weights and biases in INT8 format

## Quantization Backend

- **fbgemm**: Used for x86 CPUs (your case)
- **qnnpack**: Used for ARM CPUs
- Backend determines the optimized INT8 operations

---

## Key Insights

1. **Only Conv2d layers are quantized** - All other operations (pooling, concatenation, element-wise ops) remain FP32
2. **BatchNorm fusion** - Most BatchNorm layers are fused with Conv2d before quantization, except standalone ones (like in CoordAtt)
3. **Module wrappers** - Custom modules (BoTNet, CoordAtt) are wrapped with QAT versions that add quantization boundaries
4. **Selective quantization** - Some operations (softmax, pooling) are kept in FP32 for accuracy or numerical stability
5. **Per-channel vs Per-tensor** - Weights use per-channel quantization, activations use per-tensor quantization

