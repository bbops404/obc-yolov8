"""
Debug script to check quantization accuracy issues
Compare FP32 vs INT8 model weights and check calibration
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'obc-yolov8' / 'ultralytics10.24'))

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import ensure_module_bookkeeping

print("=" * 80)
print("Debugging Quantization Accuracy")
print("=" * 80)

# Load FP32 model
fp32_path = Path('runs/detect/train/weights/best.pt')
if not fp32_path.exists():
    # Try alternative paths
    fp32_path = Path.cwd() / 'runs/detect/train/weights/best.pt'

if fp32_path.exists():
    print(f"\n[1] Loading FP32 model from {fp32_path}...")
    fp32_checkpoint = torch.load(str(fp32_path), map_location='cpu', weights_only=False)
    if 'model' in fp32_checkpoint:
        fp32_model = fp32_checkpoint['model']
    else:
        fp32_model = fp32_checkpoint
    fp32_detection = fp32_model.model if hasattr(fp32_model, 'model') else fp32_model
    print("✓ FP32 model loaded")
else:
    print(f"⚠️  FP32 model not found at {fp32_path}")
    print("   Skipping FP32 comparison...")
    fp32_detection = None
    fp32_model = None

# Load INT8 model
int8_path = Path('runs/detect/train/weights/best_int8.pt')
if not int8_path.exists():
    int8_path = Path.cwd() / 'runs/detect/train/weights/best_int8.pt'
print(f"\n[2] Loading INT8 model from {int8_path}...")
int8_checkpoint = torch.load(str(int8_path), map_location='cpu', weights_only=False)
if 'model' in int8_checkpoint:
    int8_model = int8_checkpoint['model']
else:
    int8_model = int8_checkpoint
int8_detection = int8_model.model if hasattr(int8_model, 'model') else int8_model
ensure_module_bookkeeping(int8_detection, recursive=True)
print("✓ INT8 model loaded")

# Load QAT model to check calibration
qat_path = Path('runs/detect/train/weights/best_qat.pt')
if not qat_path.exists():
    qat_path = Path.cwd() / 'runs/detect/train/weights/best_qat.pt'
print(f"\n[3] Loading QAT model from {qat_path}...")
try:
    qat_checkpoint = torch.load(str(qat_path), map_location='cpu', weights_only=False)
    if 'model' in qat_checkpoint:
        qat_model = qat_checkpoint['model']
    else:
        qat_model = qat_checkpoint
    qat_detection = qat_model.model if hasattr(qat_model, 'model') else qat_model
    ensure_module_bookkeeping(qat_detection, recursive=True)
    print("✓ QAT model loaded")
    has_qat = True
except Exception as e:
    print(f"⚠️  Could not load QAT model: {e}")
    has_qat = False

print("\n[4] Comparing model weights...")
print("-" * 80)

# Compare first few Conv layers
from ultralytics.nn.modules.conv import Conv, Conv2, DWConv

conv_count = 0
weight_diffs = []
missing_weights = []

for name, int8_module in int8_detection.named_modules():
    if isinstance(int8_module, (Conv, Conv2, DWConv)) and hasattr(int8_module, 'conv'):
        conv_count += 1
        if conv_count > 10:  # Check first 10
            break
        
        # Get FP32 equivalent
        if fp32_detection is None:
            continue
        try:
            fp32_module = dict(fp32_detection.named_modules())[name]
        except KeyError:
            missing_weights.append(name)
            continue
        
        # Compare Conv2d weights
        if hasattr(int8_module, 'conv') and hasattr(fp32_module, 'conv'):
            int8_conv = int8_module.conv
            fp32_conv = fp32_module.conv
            
            if isinstance(int8_conv, nn.Conv2d) and isinstance(fp32_conv, nn.Conv2d):
                int8_weight = int8_conv.weight.data
                fp32_weight = fp32_conv.weight.data
                
                # Check if shapes match
                if int8_weight.shape == fp32_weight.shape:
                    # Calculate difference
                    diff = torch.abs(int8_weight - fp32_weight)
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    weight_diffs.append((name, max_diff, mean_diff))
                    
                    print(f"  {name}:")
                    print(f"    Max weight diff: {max_diff:.6f}")
                    print(f"    Mean weight diff: {mean_diff:.6f}")
                    print(f"    FP32 weight range: [{fp32_weight.min():.4f}, {fp32_weight.max():.4f}]")
                    print(f"    INT8 weight range: [{int8_weight.min():.4f}, {int8_weight.max():.4f}]")
                else:
                    print(f"  {name}: Shape mismatch! FP32: {fp32_weight.shape}, INT8: {int8_weight.shape}")

if missing_weights:
    print(f"\n⚠️  Missing weights in FP32 model: {missing_weights}")

print("\n[5] Checking for quantized Conv2d modules...")
print("-" * 80)
quantized_conv_count = 0
for name, module in int8_detection.named_modules():
    if isinstance(module, (Conv, Conv2, DWConv)) and hasattr(module, 'conv'):
        conv_layer = module.conv
        # Check if it's a quantized Conv2d
        mod_ns = getattr(type(conv_layer), '__module__', '')
        if 'torch.ao.nn.quantized.modules.conv' in mod_ns:
            quantized_conv_count += 1
            print(f"  ⚠️  Found quantized Conv2d: {name}.conv")
            if hasattr(conv_layer, '_packed_params'):
                print(f"    - Has _packed_params: True")
                print(f"    - This is a true INT8 quantized Conv")

print(f"\nTotal quantized Conv2d modules: {quantized_conv_count}")
if quantized_conv_count == 0:
    print("  ⚠️  WARNING: No quantized Conv2d modules found!")
    print("     The model might still be using FP32 Conv2d, not true INT8 quantization")
    print("     This could explain why accuracy is low - quantization might not be active")

print("\n[6] Checking FakeQuantize modules in QAT model...")
print("-" * 80)
if has_qat:
    from torch.ao.quantization import FakeQuantize
    fakequant_count = 0
    calibrated_count = 0
    uncalibrated_count = 0
    
    for name, module in qat_detection.named_modules():
        if isinstance(module, FakeQuantize):
            fakequant_count += 1
            # Check if observer has valid statistics
            if hasattr(module, 'activation_post_process'):
                observer = module.activation_post_process
                if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                    min_val = observer.min_val
                    max_val = observer.max_val
                    if min_val is not None and max_val is not None:
                        # Handle tensor values
                        if isinstance(min_val, torch.Tensor):
                            if min_val.numel() > 0:
                                min_val = min_val.item() if min_val.numel() == 1 else min_val.min().item()
                            else:
                                min_val = None
                        if isinstance(max_val, torch.Tensor):
                            if max_val.numel() > 0:
                                max_val = max_val.item() if max_val.numel() == 1 else max_val.max().item()
                            else:
                                max_val = None
                        
                        if min_val is not None and max_val is not None:
                            if max_val > min_val:
                                calibrated_count += 1
                            else:
                                uncalibrated_count += 1
                                if uncalibrated_count <= 5:
                                    print(f"  ⚠️  Uncalibrated: {name} (min={min_val}, max={max_val})")
                        else:
                            uncalibrated_count += 1
                    else:
                        uncalibrated_count += 1
    
    print(f"  Total FakeQuantize modules: {fakequant_count}")
    print(f"  Calibrated (valid stats): {calibrated_count}")
    print(f"  Uncalibrated (invalid stats): {uncalibrated_count}")
    
    if uncalibrated_count > 0:
        print(f"\n  ⚠️  WARNING: {uncalibrated_count} FakeQuantize modules are not calibrated!")
        print("     This means quantization parameters are not set correctly")
        print("     The model needs proper calibration during QAT training")

print("\n[7] Testing forward pass with dummy input...")
print("-" * 80)
try:
    dummy_input = torch.randn(1, 3, 640, 640)
    
    def _extract_prediction(tensor_or_tuple):
        if isinstance(tensor_or_tuple, torch.Tensor):
            return tensor_or_tuple
        if isinstance(tensor_or_tuple, (list, tuple)):
            for item in tensor_or_tuple:
                if isinstance(item, torch.Tensor):
                    return item
        return None

    fp32_detection.eval() if fp32_detection is not None else None
    int8_detection.eval()

    with torch.no_grad():
        fp32_out = fp32_detection(dummy_input) if fp32_detection is not None else None
        int8_out = int8_detection(dummy_input)

    if fp32_detection is not None:
        print("  ✓ FP32 forward pass succeeded")
    else:
        print("  ⚠️  Skipping FP32 forward pass (model not loaded)")
    
    print("  ✓ INT8 forward pass succeeded")
    
    fp32_tensor = _extract_prediction(fp32_out)
    int8_tensor = _extract_prediction(int8_out)
            
    if fp32_tensor is not None and int8_tensor is not None and fp32_tensor.shape == int8_tensor.shape:
                    output_diff = torch.abs(fp32_tensor - int8_tensor)
                    max_output_diff = output_diff.max().item()
                    mean_output_diff = output_diff.mean().item()
        print("\n  Output comparison:")
                    print(f"    Max difference: {max_output_diff:.6f}")
                    print(f"    Mean difference: {mean_output_diff:.6f}")
                    print(f"    FP32 output range: [{fp32_tensor.min():.4f}, {fp32_tensor.max():.4f}]")
                    print(f"    INT8 output range: [{int8_tensor.min():.4f}, {int8_tensor.max():.4f}]")
                    if max_output_diff > 10.0:
            print("    ⚠️  WARNING: Large output difference! Quantization may be too aggressive")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if quantized_conv_count == 0:
    print("❌ CRITICAL: No quantized Conv2d modules found!")
    print("   The model is likely still using FP32 Conv2d, not true INT8 quantization")
    print("   This means quantization didn't work properly during conversion")
    print("\n   Possible causes:")
    print("   1. Conversion didn't actually quantize the Conv2d layers")
    print("   2. The model structure prevents quantization")
    print("   3. qconfig wasn't set properly during prepare_for_qat()")
elif uncalibrated_count > 0 if has_qat else False:
    print("⚠️  WARNING: Many FakeQuantize modules are not calibrated")
    print("   Quantization parameters (scale/zero_point) are incorrect")
    print("   This causes massive accuracy loss")
else:
    print("✓ Model structure looks correct")
    print("⚠️  But accuracy is very low - may need more QAT training epochs")

print("=" * 80)

