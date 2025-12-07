#!/usr/bin/env python3
"""
Check which model type is actually being evaluated by evaluate_int8.py
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, 'obc-yolov8/ultralytics10.24')
from ultralytics import YOLO
from torch.ao.quantization import FakeQuantize

print("=" * 80)
print("Checking Evaluation Model Type")
print("=" * 80)

# Simulate what evaluate_int8.py does
int8_path = "runs/detect/qat_fold3_qnnpack3/weights/last_int8.pt"
qat_path = "runs/detect/qat_fold3_qnnpack3/weights/last_qat.pt"

print(f"\n1. Loading INT8 checkpoint: {int8_path}")
try:
    int8_ckpt = torch.load(int8_path, map_location='cpu', weights_only=False)
    int8_model_obj = int8_ckpt.get('model')
    
    # Check if it's actually quantized
    is_quantized = False
    has_fakequant = False
    
    try:
        for name, module in int8_model_obj.named_modules():
            if isinstance(module, FakeQuantize):
                has_fakequant = True
                break
            elif hasattr(module, '_packed_params') and module._packed_params is not None:
                is_quantized = True
                break
    except AttributeError:
        # QuantizedConv2d doesn't have _modules
        is_quantized = True
    
    print(f"   Has FakeQuantize: {has_fakequant}")
    print(f"   Has quantized modules: {is_quantized}")
    
    if has_fakequant:
        print("   ⚠️  INT8 model still has FakeQuantize (not fully converted)")
    elif is_quantized:
        print("   ✓ INT8 model is quantized (no FakeQuantize)")
    else:
        print("   ⚠️  INT8 model appears to be FP32")
        
except Exception as e:
    print(f"   ❌ Error loading INT8: {e}")
    int8_model_obj = None

print(f"\n2. Checking QAT fallback path: {qat_path}")
if Path(qat_path).exists():
    try:
        qat_ckpt = torch.load(qat_path, map_location='cpu', weights_only=False)
        qat_model_obj = qat_ckpt.get('model')
        
        fakequant_count = 0
        try:
            for name, module in qat_model_obj.named_modules():
                if isinstance(module, FakeQuantize):
                    fakequant_count += 1
        except:
            pass
        
        print(f"   QAT model has {fakequant_count} FakeQuantize modules")
        print(f"   ✓ QAT model available for fallback")
    except Exception as e:
        print(f"   ❌ Error loading QAT: {e}")
        qat_model_obj = None
else:
    print(f"   ⚠️  QAT checkpoint not found")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("\nWhy evaluate_int8.py produces results:")
print("\n1. Dummy Forward Pass Provides Minimal Statistics:")
print("   - Line 1411-1417 in tasks.py runs a dummy forward pass")
print("   - This calibrates observers with a SINGLE dummy input (3x3 image)")
print("   - Provides min/max values to prevent NaN during conversion")
print("   - BUT: Statistics are based on random dummy data, not real training data")
print("   - Quality: Poor (single sample, random values)")
print("\n2. Conversion Succeeds (No NaN):")
print("   - Dummy forward pass prevents NaN errors during from_float()")
print("   - INT8 model is created successfully")
print("   - BUT: Quantization parameters are suboptimal")
print("\n3. Evaluation May Work OR Fall Back:")
print("   - If INT8 evaluation works: Uses poorly calibrated INT8 model")
print("   - If INT8 evaluation fails: Falls back to QAT model (lines 701-757)")
print("   - QAT model has proper statistics from training")
print("\n4. The Log Shows:")
print("   - Line 144252: 'INT8 evaluation failed: cannot convert float NaN to integer'")
print("   - This happens during training-time evaluation")
print("   - evaluate_int8.py may handle this differently (fallback or better error handling)")
print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("The results you see are likely:")
print("  A) INT8 model with poor statistics (dummy-calibrated)")
print("  B) QAT model fallback (if INT8 evaluation fails)")
print("\nTo verify, check the evaluate_int8.py output for:")
print("  - 'Falling back to QAT model' message")
print("  - 'Note: Using QAT model due to INT8 backend limitation'")
print("=" * 80)
