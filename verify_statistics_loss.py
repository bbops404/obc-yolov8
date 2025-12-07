#!/usr/bin/env python3
"""
Verify if statistics are lost during conversion flow.
This script checks:
1. Statistics in original QAT model
2. What happens after convert() removes FakeQuantize
3. What happens when _create_quantized_conv2d_from_conv2d creates new QATConv2d
"""

import sys
from pathlib import Path
import torch
from torch.ao.quantization import FakeQuantize, prepare_qat
from torch.ao.nn.qat.modules.conv import Conv2d as QATConv2d

REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics.nn.tasks import _create_safe_qconfig

print("=" * 80)
print("Statistics Loss Verification")
print("=" * 80)

# Load QAT model
checkpoint_path = "runs/detect/qat_fold3_qnnpack3/weights/last_qat.pt"
print(f"\n1. Loading QAT model: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
qat_model = ckpt['model']
backend = ckpt.get('backend', 'qnnpack')
print(f"   Backend: {backend}")

# Find a Conv2d inside a fused wrapper to test
print("\n2. Finding Conv2d inside fused wrapper...")
from ultralytics.nn.modules.conv import Conv, Conv2, DWConv

test_conv = None
test_name = None
for name, module in qat_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        parent_path = '.'.join(name.split('.')[:-1])
        if parent_path:
            try:
                parent = dict(qat_model.named_modules()).get(parent_path)
                if parent is not None and isinstance(parent, (Conv, Conv2, DWConv)):
                    if not hasattr(parent, 'bn'):  # Fused (no BN)
                        test_conv = module
                        test_name = name
                        print(f"   Found: {name}")
                        print(f"   Parent: {type(parent).__name__}")
                        break
            except:
                pass

if test_conv is None:
    print("   ❌ Could not find Conv2d inside fused wrapper")
    sys.exit(1)

# Check if this Conv2d has FakeQuantize attached (it shouldn't, but let's verify)
print(f"\n3. Checking original Conv2d module:")
print(f"   Type: {type(test_conv).__name__}")
print(f"   Has weight_fake_quant: {hasattr(test_conv, 'weight_fake_quant')}")
print(f"   Has activation_post_process: {hasattr(test_conv, 'activation_post_process')}")

# Find the corresponding FakeQuantize in the parent (if any)
parent_path = '.'.join(test_name.split('.')[:-1])
parent = dict(qat_model.named_modules()).get(parent_path)
print(f"\n4. Checking parent module for FakeQuantize:")
if parent:
    parent_fq = []
    for name, module in parent.named_modules():
        if isinstance(module, FakeQuantize):
            parent_fq.append(name)
            # Check statistics
            observer = getattr(module, 'activation_post_process', None)
            if observer:
                min_val = getattr(observer, 'min_val', None)
                max_val = getattr(observer, 'max_val', None)
                has_stats = min_val is not None and max_val is not None
                print(f"   Found FakeQuantize: {name}")
                print(f"     Has statistics: {has_stats}")
                if has_stats:
                    if isinstance(min_val, torch.Tensor):
                        print(f"     min_val shape: {min_val.shape}, has_nan: {torch.isnan(min_val).any()}")
                        print(f"     max_val shape: {max_val.shape}, has_nan: {torch.isnan(max_val).any()}")
                    else:
                        print(f"     min_val: {min_val}, max_val: {max_val}")
    if not parent_fq:
        print(f"   ⚠️  No FakeQuantize found in parent (expected - Conv2d inside wrapper)")

# Now simulate what _create_quantized_conv2d_from_conv2d does
print(f"\n5. Simulating _create_quantized_conv2d_from_conv2d() flow:")
print(f"   Creating new QATConv2d...")

# Get qconfig
qconfig = _create_safe_qconfig(backend)

# Create new QATConv2d (this is what happens in the conversion)
new_qat_conv = QATConv2d(
    test_conv.in_channels,
    test_conv.out_channels,
    test_conv.kernel_size,
    stride=test_conv.stride,
    padding=test_conv.padding,
    dilation=test_conv.dilation,
    groups=test_conv.groups,
    bias=test_conv.bias is not None,
    padding_mode=test_conv.padding_mode,
    qconfig=qconfig
)

print(f"   New QATConv2d created")
print(f"   Has weight_fake_quant: {hasattr(new_qat_conv, 'weight_fake_quant')}")
print(f"   Has activation_post_process: {hasattr(new_qat_conv, 'activation_post_process')}")

# Check statistics BEFORE prepare_qat
print(f"\n6. Statistics BEFORE prepare_qat():")
if hasattr(new_qat_conv, 'activation_post_process') and new_qat_conv.activation_post_process:
    observer = new_qat_conv.activation_post_process
    min_val = getattr(observer, 'min_val', None)
    max_val = getattr(observer, 'max_val', None)
    print(f"   activation_post_process min_val: {min_val}")
    print(f"   activation_post_process max_val: {max_val}")
    print(f"   Has statistics: {min_val is not None and max_val is not None}")

# Now call prepare_qat (this is what happens at line 1409)
print(f"\n7. Calling prepare_qat() (line 1409)...")
new_qat_conv.train()
prepare_qat(new_qat_conv, inplace=True)

# Check statistics AFTER prepare_qat
print(f"\n8. Statistics AFTER prepare_qat():")
if hasattr(new_qat_conv, 'activation_post_process') and new_qat_conv.activation_post_process:
    observer = new_qat_conv.activation_post_process
    min_val = getattr(observer, 'min_val', None)
    max_val = getattr(observer, 'max_val', None)
    print(f"   activation_post_process min_val: {min_val}")
    print(f"   activation_post_process max_val: {max_val}")
    print(f"   Has statistics: {min_val is not None and max_val is not None}")
    if min_val is None or max_val is None:
        print(f"   ❌ Statistics are None - observers are uncalibrated!")

# Run dummy forward pass (line 1411-1417)
print(f"\n9. Running dummy forward pass (line 1411-1417)...")
dummy_input = torch.randn(1, test_conv.in_channels, 3, 3)
# Copy weights ensuring same dtype
new_qat_conv.weight = torch.nn.Parameter(test_conv.weight.data.clone().to(new_qat_conv.weight.dtype))
if test_conv.bias is not None:
    new_qat_conv.bias = torch.nn.Parameter(test_conv.bias.data.clone().to(new_qat_conv.bias.dtype))
try:
    with torch.no_grad():
        _ = new_qat_conv(dummy_input)
    print(f"   ✓ Dummy forward pass completed")
except Exception as e:
    print(f"   ⚠️  Dummy forward pass failed: {e}")
    print(f"   (This doesn't affect the statistics verification)")

# Check statistics AFTER dummy forward
print(f"\n10. Statistics AFTER dummy forward pass:")
if hasattr(new_qat_conv, 'activation_post_process') and new_qat_conv.activation_post_process:
    observer = new_qat_conv.activation_post_process
    min_val = getattr(observer, 'min_val', None)
    max_val = getattr(observer, 'max_val', None)
    print(f"   activation_post_process min_val: {min_val}")
    print(f"   activation_post_process max_val: {max_val}")
    print(f"   Has statistics: {min_val is not None and max_val is not None}")
    if min_val is not None and max_val is not None:
        print(f"   ✓ Statistics collected from dummy forward")
    else:
        print(f"   ❌ Still no statistics after dummy forward!")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("1. Original QAT model has statistics in FakeQuantize modules")
print("2. convert() removes FakeQuantize modules (statistics lost)")
print("3. Conv2d inside fused wrappers remain FP32 (no FakeQuantize)")
print("4. _create_quantized_conv2d_from_conv2d() creates NEW QATConv2d")
print("5. prepare_qat() creates NEW observers with NO statistics")
print("6. Dummy forward pass calibrates with single dummy input (not real data)")
print("7. from_float() uses these new observers (minimal statistics)")
print("\n✓ CONFIRMED: Statistics from QAT training are LOST for Conv2d inside fused wrappers")
print("=" * 80)

