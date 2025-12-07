#!/usr/bin/env python3
"""Simple script to check QAT statistics."""
import sys
from pathlib import Path
import torch
from torch.ao.quantization import FakeQuantize

REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/qat_fold3_qnnpack3/weights/last_qat.pt"

print("=" * 80)
print("QAT Statistics Checker")
print("=" * 80)
print(f"Loading: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
print(f"✓ Loaded checkpoint")

if isinstance(checkpoint, dict):
    print(f"  Keys: {list(checkpoint.keys())}")
    if 'qat' in checkpoint:
        print(f"  QAT flag: {checkpoint['qat']}")
    if 'backend' in checkpoint:
        print(f"  Backend: {checkpoint['backend']}")

model = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
print(f"  Model type: {type(model).__name__}")

# Count FakeQuantize modules
fakequant_count = 0
uncalibrated = []
calibrated = []
has_nan = []

def check_module(name, module):
    global fakequant_count, uncalibrated, calibrated, has_nan
    if isinstance(module, FakeQuantize):
        fakequant_count += 1
        try:
            observer = getattr(module, 'activation_post_process', None)
            if observer is None:
                # Check scale/zero_point directly
                scale = getattr(module, 'scale', None)
                zero_point = getattr(module, 'zero_point', None)
                if scale is None or zero_point is None:
                    uncalibrated.append(name)
                elif isinstance(scale, torch.Tensor) and (torch.isnan(scale).any() or torch.isnan(zero_point).any()):
                    has_nan.append(name)
                else:
                    calibrated.append(name)
            else:
                min_val = getattr(observer, 'min_val', None)
                max_val = getattr(observer, 'max_val', None)
                if min_val is None or max_val is None:
                    uncalibrated.append(name)
                elif isinstance(min_val, torch.Tensor):
                    if torch.isnan(min_val).any() or torch.isnan(max_val).any():
                        has_nan.append(name)
                    else:
                        calibrated.append(name)
                else:
                    calibrated.append(name)
        except Exception as e:
            uncalibrated.append(f"{name} (error: {e})")

# Try to iterate modules safely
try:
    for name, module in model.named_modules():
        check_module(name, module)
except Exception as e:
    print(f"  ⚠️  Error with named_modules: {e}")
    print("  Trying recursive search...")
    def find_fq(module, path=""):
        results = []
        if isinstance(module, FakeQuantize):
            check_module(path, module)
        if hasattr(module, '_modules') and module._modules is not None:
            for n, m in module._modules.items():
                if m is not None:
                    new_path = f"{path}.{n}" if path else n
                    find_fq(m, new_path)
    try:
        find_fq(model)
    except Exception as e2:
        print(f"  ❌ Recursive search failed: {e2}")

print(f"\nStatistics Summary:")
print(f"  Total FakeQuantize modules: {fakequant_count}")
print(f"  ✓ Calibrated: {len(calibrated)}")
print(f"  ❌ Uncalibrated: {len(uncalibrated)}")
print(f"  ⚠️  Has NaN: {len(has_nan)}")

if has_nan:
    print(f"\n  ⚠️  Modules with NaN (first 5):")
    for name in has_nan[:5]:
        print(f"      - {name}")

if uncalibrated:
    print(f"\n  ❌ Uncalibrated modules (first 5):")
    for name in uncalibrated[:5]:
        print(f"      - {name}")

print("\n" + "=" * 80)
if len(has_nan) == 0 and len(uncalibrated) == 0 and fakequant_count > 0:
    print("✓ Statistics check PASSED - all modules calibrated")
elif fakequant_count == 0:
    print("⚠️  No FakeQuantize modules found (may be INT8 model)")
else:
    print("❌ Statistics check FAILED")
print("=" * 80)


