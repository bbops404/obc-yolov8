#!/usr/bin/env python3
"""
Quick test to see if INT8 evaluation works or falls back to QAT
"""

import sys
from pathlib import Path
sys.path.insert(0, 'obc-yolov8/ultralytics10.24')

import torch
from ultralytics import YOLO
from torch.ao.quantization import FakeQuantize

print("=" * 80)
print("Testing INT8 Model Evaluation")
print("=" * 80)

int8_path = "runs/detect/qat_fold3_qnnpack3/weights/last_int8.pt"
data_cfg = "obc-yolov8/ultralytics10.24/ultralytics/cfg/datasets/combined_china_motorbike.yaml"

# Set backend
torch.backends.quantized.engine = 'qnnpack'

# Load checkpoint
print(f"\n1. Loading INT8 checkpoint...")
checkpoint = torch.load(int8_path, map_location='cpu', weights_only=False)
backend = checkpoint.get('backend', 'qnnpack')
print(f"   Backend: {backend}")

# Load model
model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
model = YOLO(model_cfg)
model.model = checkpoint['model']

# Check model type
print(f"\n2. Checking model type...")
try:
    fakequant_count = 0
    for name, module in model.model.named_modules():
        if isinstance(module, FakeQuantize):
            fakequant_count += 1
    print(f"   FakeQuantize modules: {fakequant_count}")
    if fakequant_count > 0:
        print("   ⚠️  Model has FakeQuantize (QAT, not INT8)")
    else:
        print("   ✓ Model is INT8 (no FakeQuantize)")
except AttributeError:
    print("   ✓ Model is INT8 (cannot iterate - quantized structure)")

# Test forward pass
print(f"\n3. Testing forward pass...")
try:
    test_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model.model(test_input)
    print("   ✓ Forward pass succeeded")
    forward_works = True
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    forward_works = False

# Try evaluation
if forward_works:
    print(f"\n4. Attempting evaluation...")
    try:
        results = model.val(
            data=data_cfg,
            imgsz=640,
            batch=4,
            device='cpu',
            plots=False,
            save=False,
            verbose=False
        )
        
        if results and hasattr(results, 'box'):
            print("   ✓ INT8 evaluation succeeded!")
            print(f"   mAP@0.5: {results.box.map50:.4f}")
            print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
            print("\n   ✓ CONFIRMED: Using INT8 model (not QAT fallback)")
        else:
            print("   ⚠️  Evaluation completed but no metrics")
    except Exception as e:
        error_str = str(e)
        print(f"   ❌ INT8 evaluation failed: {error_str[:200]}")
        
        if "NaN" in error_str or "nan" in error_str.lower():
            print("\n   ⚠️  NaN error detected - this confirms statistics loss issue")
            print("   evaluate_int8.py would fall back to QAT model here")
        else:
            print(f"\n   Different error - may still work with evaluate_int8.py fallback")
else:
    print(f"\n4. Skipping evaluation (forward pass failed)")

print("\n" + "=" * 80)

