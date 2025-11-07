"""
Diagnostic script to check QAT preparation
Verifies if FakeQuantize modules are actually being inserted
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'obc-yolov8' / 'ultralytics10.24'))

import torch
import torch.nn as nn
from ultralytics import YOLO
from torch.ao.quantization import FakeQuantize

print("=" * 80)
print("QAT Diagnostic Script")
print("=" * 80)

# Load model
print("\n1. Loading model...")
model = YOLO('obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml')
print(f"   Model loaded: {type(model.model)}")
print(f"   Number of layers: {len(list(model.model.model))}")

# Prepare for QAT
print("\n2. Preparing model for QAT...")
example_input = torch.randn(1, 3, 640, 640)
model_prepared = model.model.prepare_for_qat(backend='fbgemm', example_input=example_input)

# Check FakeQuantize modules
print("\n3. Checking FakeQuantize modules...")
fakequant_modules = [(n, m) for n, m in model_prepared.named_modules() if isinstance(m, FakeQuantize)]
print(f"   Total FakeQuantize modules found: {len(fakequant_modules)}")

if fakequant_modules:
    print("\n   First 15 FakeQuantize modules:")
    for name, module in fakequant_modules[:15]:
        print(f"     - {name}")
else:
    print("   ⚠️ WARNING: No FakeQuantize modules found!")

# Check qconfig on Conv layers
print("\n4. Checking qconfig on Conv2d layers...")
conv_layers = [(n, m) for n, m in model_prepared.named_modules() if isinstance(m, nn.Conv2d)]
print(f"   Total Conv2d layers: {len(conv_layers)}")

conv_with_qconfig = [n for n, m in conv_layers if hasattr(m, 'qconfig') and m.qconfig is not None]
print(f"   Conv2d with qconfig: {len(conv_with_qconfig)}")

if conv_with_qconfig:
    print("\n   First 10 Conv2d layers with qconfig:")
    for name in conv_with_qconfig[:10]:
        print(f"     - {name}")
else:
    print("   ⚠️ WARNING: No Conv2d layers have qconfig!")

# Check model.model (Sequential) qconfig
print("\n5. Checking qconfig on model structure...")
print(f"   model has qconfig: {hasattr(model_prepared, 'qconfig')}")
print(f"   model.model has qconfig: {hasattr(model_prepared.model, 'qconfig')}")

for i, layer in enumerate(model_prepared.model[:5]):
    has_qconfig = hasattr(layer, 'qconfig') and layer.qconfig is not None
    print(f"   Layer {i} ({type(layer).__name__}): has_qconfig={has_qconfig}")

# Try conversion
print("\n6. Testing conversion...")
model_prepared.eval()

try:
    from torch.ao.quantization import convert
    model_int8 = convert(model_prepared, inplace=False)
    print("   ✓ Conversion succeeded")
    
    # Check for quantized modules
    quantized_modules = []
    for name, module in model_int8.named_modules():
        module_type = type(module).__name__
        if 'Quantized' in module_type or hasattr(module, '_packed_params'):
            quantized_modules.append((name, module_type))
    
    print(f"   Quantized modules found: {len(quantized_modules)}")
    if quantized_modules:
        print("\n   First 10 quantized modules:")
        for name, mod_type in quantized_modules[:10]:
            print(f"     - {name}: {mod_type}")
    else:
        print("   ⚠️ WARNING: No quantized modules found after conversion!")
        
except Exception as e:
    print(f"   ✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Diagnostic Summary")
print("=" * 80)
print(f"FakeQuantize modules: {len(fakequant_modules)}")
print(f"Conv2d with qconfig: {len(conv_with_qconfig)}")
print(f"Quantized modules after convert: {len(quantized_modules) if 'quantized_modules' in locals() else 0}")
print("=" * 80)

if len(fakequant_modules) == 0:
    print("\n⚠️ ISSUE: No FakeQuantize modules inserted during prepare_qat()")
    print("   This means QAT preparation is not working.")
    print("   Likely causes:")
    print("   - qconfig not propagating to leaf modules")
    print("   - Model structure incompatible with eager mode QAT")
    print("   - Need to fuse modules first (Conv+BN)")

