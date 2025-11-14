# Quantization Library Used

## Overview

This project uses **PyTorch's built-in quantization library** (`torch.ao.quantization`) for both QAT (Quantization-Aware Training) and PTQ (Post-Training Quantization).

---

## Library: PyTorch AO (torch.ao.quantization)

### What is torch.ao.quantization?

`torch.ao.quantization` is PyTorch's official quantization framework (formerly `torch.quantization`). The "ao" stands for "AO" (the new namespace introduced in PyTorch 1.8+). It provides:

- **QAT (Quantization-Aware Training)**: Train models with quantization simulation
- **PTQ (Post-Training Quantization)**: Quantize pre-trained models without retraining
- **FX Graph Mode**: Graph-based quantization for better accuracy
- **Eager Mode**: Module-based quantization for compatibility

---

## Key Components Used

### 1. **FakeQuantize Modules**
```python
from torch.ao.quantization.fake_quantize import FakeQuantize
```

**Purpose**: Simulates quantization during training without actually quantizing weights.

**How it works**:
- During QAT training: Uses FP32 weights but applies quantization noise
- Collects statistics (min/max values) through observers
- After training: Statistics are used to create real quantized weights

**Example from code**:
```python
activation_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine
)
```

### 2. **QuantStub and DeQuantStub**
```python
from torch.ao.quantization import QuantStub, DeQuantStub
```

**Purpose**: Define quantization boundaries in custom modules.

**Usage**:
- `QuantStub()`: Marks where quantization starts (input boundary)
- `DeQuantStub()`: Marks where quantization ends (output boundary)

**Example from QATCoordAtt**:
```python
class QATCoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        self.quant = QuantStub()      # Input quantization
        self.coordatt = CoordAtt(...)
        self.dequant = DeQuantStub()  # Output dequantization
```

### 3. **QConfig and QConfigDict**
```python
from torch.ao.quantization import get_default_qat_qconfig
```

**Purpose**: Configuration that specifies how to quantize each layer.

**Components**:
- **Weight quantization**: Per-channel INT8 (qint8)
- **Activation quantization**: Per-tensor INT8 (quint8)
- **Observer**: Collects statistics (MinMaxObserver, MovingAverageMinMaxObserver)

**Example**:
```python
qconfig = get_default_qat_qconfig(backend='fbgemm')
qconfig_dict = {
    "": qconfig,  # Default for all modules
    "module_name": {
        "model.25.dfl": None  # Exclude DFL from quantization
    }
}
```

### 4. **Prepare Functions**

#### FX Graph Mode (Preferred)
```python
from torch.ao.quantization.quantize_fx import prepare_qat_fx
```

**Purpose**: Graph-based quantization using FX (Functional Transform).

**Advantages**:
- Better accuracy (sees full computation graph)
- Handles complex operations
- More flexible

**Usage**:
```python
model_prepared = prepare_qat_fx(
    model,
    qconfig_dict,
    example_inputs=(example_input,),
    backend_config=None
)
```

#### Eager Mode (Fallback)
```python
from torch.ao.quantization import prepare_qat
```

**Purpose**: Module-based quantization (legacy method).

**When used**: When FX mode fails or for compatibility.

**Usage**:
```python
model_prepared = prepare_qat(model, inplace=False)
```

### 5. **Convert Function**
```python
from torch.ao.quantization import convert
```

**Purpose**: Converts QAT model (with FakeQuantize) to real INT8 model.

**What it does**:
- Replaces `FakeQuantize` with real quantized operations
- Converts `nn.Conv2d` → `torch.ao.nn.quantized.modules.conv.Conv2d`
- Uses collected statistics to create quantized weights

**Usage**:
```python
model_int8 = model.convert_to_quantized()
# Internally calls: convert(model)
```

---

## Quantization Backends

### 1. **fbgemm** (Facebook GEMM)
- **Target**: x86 CPUs (Intel/AMD)
- **Optimized for**: Desktop and server CPUs
- **Your model uses**: `backend='fbgemm'`

### 2. **qnnpack** (Quantized Neural Network PACK)
- **Target**: ARM CPUs (mobile devices)
- **Optimized for**: Mobile and embedded devices

**Setting backend**:
```python
torch.backends.quantized.engine = 'fbgemm'  # or 'qnnpack'
```

**Important**: Backend must be set **BEFORE** creating any quantized operations!

---

## Quantization Workflow

### QAT (Quantization-Aware Training) Workflow

1. **Prepare Model**:
   ```python
   model = model.prepare_for_qat(backend='fbgemm', example_input=example_input)
   ```
   - Inserts `FakeQuantize` modules
   - Attaches observers to collect statistics
   - Model still uses FP32 weights

2. **Train Model**:
   ```python
   trainer.train()  # Normal training loop
   ```
   - Model trains with quantization simulation
   - Observers collect min/max statistics
   - Weights adapt to quantization noise

3. **Convert to INT8**:
   ```python
   model_int8 = model.convert_to_quantized()
   ```
   - Replaces `FakeQuantize` with real quantized ops
   - Creates `torch.ao.nn.quantized.modules.conv.Conv2d`
   - Weights become INT8 with scale/zero_point

### PTQ (Post-Training Quantization) Workflow

1. **Prepare Model**:
   ```python
   model = model.prepare_for_ptq(backend='fbgemm')
   ```
   - Inserts observers (not FakeQuantize)
   - Model remains FP32

2. **Calibrate**:
   ```python
   model = model.calibrate_ptq(calibration_data, num_batches=100)
   ```
   - Runs calibration data through model
   - Observers collect statistics
   - No training, just statistics collection

3. **Convert to INT8**:
   ```python
   model_int8 = model.convert_ptq_to_int8(backend='fbgemm')
   ```
   - Uses collected statistics to quantize
   - Creates real INT8 operations

---

## Quantized Module Types

After conversion, modules become:

### Quantized Conv2d
```python
torch.ao.nn.quantized.modules.conv.Conv2d
```

**Properties**:
- `scale`: Quantization scale factor (float)
- `zero_point`: Quantization zero point (int)
- `_packed_params`: Packed INT8 weights and biases
- Module path: `torch.ao.nn.quantized.modules.conv`

**Example from your model**:
```python
# Before conversion
model.0.conv  # nn.Conv2d

# After conversion
model.0.conv  # torch.ao.nn.quantized.modules.conv.Conv2d
# Has: scale=0.59553689, zero_point=68, _packed_params=True
```

---

## Code Locations

### Main Quantization Code

1. **QAT Preparation**: `obc-yolov8/ultralytics10.24/ultralytics/nn/tasks.py`
   - `prepare_for_qat()` method (line ~625)
   - Uses FX Graph Mode with Eager Mode fallback

2. **QAT Conversion**: `obc-yolov8/ultralytics10.24/ultralytics/nn/tasks.py`
   - `convert_to_quantized()` method (line ~820)
   - Converts FakeQuantize to real INT8 ops

3. **PTQ Preparation**: `obc-yolov8/ultralytics10.24/ultralytics/nn/tasks.py`
   - `prepare_for_ptq()` method (line ~1921)
   - Inserts observers for calibration

4. **PTQ Conversion**: `obc-yolov8/ultralytics10.24/ultralytics/nn/tasks.py`
   - `convert_ptq_to_int8()` method (line ~2451)
   - Converts calibrated model to INT8

5. **QAT Modules**: `obc-yolov8/ultralytics10.24/ultralytics/nn/qat_modules.py`
   - Custom QAT wrappers (QATBoTNet, QATCoordAtt)
   - Uses QuantStub/DeQuantStub for boundaries

### Training Scripts

1. **QAT Training**: `train_qat.py`
   - Entry point for QAT training
   - Calls `prepare_for_qat()` and `convert_to_quantized()`

2. **PTQ Training**: `train_ptq.py`
   - Entry point for PTQ
   - Calls `prepare_for_ptq()`, `calibrate_ptq()`, `convert_ptq_to_int8()`

---

## Key Features Used

### 1. **Hybrid FX + Eager Mode**
- FX Graph Mode for most of the model
- Eager Mode for complex modules (Detect head, tensor ops)
- Best of both worlds: accuracy + compatibility

### 2. **Selective Quantization**
- Can exclude specific modules (ODConv, DFL)
- Uses `qconfig_dict` to control what gets quantized

### 3. **Conv+BN+Activation Fusion**
- Fuses layers before quantization
- Creates `QuantizedConvReLU2d` (fused operation)
- Better performance and accuracy

### 4. **Custom QAT Modules**
- QATBoTNet: Wraps BoTNet with quantization boundaries
- QATCoordAtt: Wraps CoordAtt with quantization boundaries
- Maintains module structure while enabling quantization

---

## Quantization Parameters

Each quantized layer has:

1. **Scale** (float): Maps quantized integers to real values
   - Formula: `real_value = (quantized_value - zero_point) * scale`

2. **Zero Point** (int): Offset for quantization
   - Typically 0-255 for quint8 activations
   - Typically -128 to 127 for qint8 weights

3. **_packed_params**: Packed INT8 weights and biases
   - Optimized format for fast INT8 operations
   - Contains quantized weight tensor and bias

---

## Summary

**Library**: PyTorch's `torch.ao.quantization`

**Key APIs**:
- `prepare_qat_fx()` / `prepare_qat()` - Prepare for QAT
- `prepare_ptq()` - Prepare for PTQ
- `convert()` - Convert to INT8
- `FakeQuantize` - Simulate quantization during training
- `QuantStub` / `DeQuantStub` - Quantization boundaries

**Backend**: `fbgemm` (for x86 CPUs)

**Quantized Modules**: `torch.ao.nn.quantized.modules.conv.Conv2d`

**Result**: 66 quantized Conv2d layers with INT8 weights and activations

---

## References

- PyTorch Quantization Documentation: https://pytorch.org/docs/stable/quantization.html
- FX Graph Mode: https://pytorch.org/docs/stable/fx.html
- QAT Tutorial: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html

