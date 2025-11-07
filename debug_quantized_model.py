"""
Debug script to inspect quantized INT8 model structure.

This script helps identify:
1. Module types (QuantizedConv2d vs regular Conv2d)
2. Which operations fail (named_modules, .eval(), .cpu(), etc.)
3. Count quantized vs FP32 modules
4. Verify model structure after conversion
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'obc-yolov8' / 'ultralytics10.24'))

import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize
from ultralytics import YOLO

# QuantizedConv2d is not directly importable - it's created dynamically by convert()
# We'll check for it by string matching module type names

print("=" * 80)
print("Quantized Model Structure Debug")
print("=" * 80)

# Configuration
int8_checkpoint_path = 'runs/detect/train/weights/best_int8.pt'
imgsz = 640

# ============================================================================
# Step 1: Load INT8 checkpoint and inspect metadata
# ============================================================================
print("\n[Step 1] Loading INT8 checkpoint...")
if not Path(int8_checkpoint_path).exists():
    print(f"   ✗ Checkpoint not found: {int8_checkpoint_path}")
    print("   Please run train_qat.py first to generate the INT8 model.")
    sys.exit(1)

try:
    checkpoint = torch.load(int8_checkpoint_path, map_location='cpu', weights_only=False)
    backend = checkpoint.get('backend', 'unknown')
    is_int8 = checkpoint.get('int8', False)
    print(f"   ✓ Checkpoint loaded")
    print(f"   Backend: {backend}")
    print(f"   INT8 flag: {is_int8}")
    
    if 'model' in checkpoint:
        model_obj = checkpoint['model']
        if isinstance(model_obj, dict):
            print("   Model format: state_dict")
        else:
            print("   Model format: full model object")
            print(f"   Model type: {type(model_obj).__name__}")
except Exception as e:
    print(f"   ✗ Failed to load checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Step 2: Set quantization backend and load model
# ============================================================================
print("\n[Step 2] Setting quantization backend and loading model...")
try:
    if backend in ['fbgemm', 'qnnpack']:
        torch.backends.quantized.engine = backend
        print(f"   ✓ Set quantization backend engine to {backend}")
    
    model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
    model = YOLO(model_cfg)
    
    if isinstance(model_obj, dict):
        # Load from state_dict
        print("   Loading from state_dict...")
        model.model = model.model.prepare_for_qat(
            backend=backend,
            example_input=torch.randn(1, 3, imgsz, imgsz)
        )
        model.model = model.model.convert_to_quantized()
        model.model.load_state_dict(model_obj, strict=False)
    else:
        # Full model object
        print("   Loading full model object...")
        model.model = model_obj
    
    print("   ✓ Model loaded")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Step 3: Inspect module types
# ============================================================================
print("\n[Step 3] Inspecting module types...")
module_types = {}
quantized_modules = []
fp32_modules = []
has_packed_params = []

try:
    for name, module in model.model.named_modules():
        module_type = type(module).__name__
        module_types[module_type] = module_types.get(module_type, 0) + 1
        
        # Check for quantized modules
        if 'Quantized' in module_type:
            quantized_modules.append((name, module_type))
        elif hasattr(module, '_packed_params'):
            has_packed_params.append((name, module_type))
            quantized_modules.append((name, module_type))
        elif isinstance(module, nn.Conv2d) and 'Quantized' not in type(module).__name__:
            fp32_modules.append((name, module_type))
    
    print(f"   Total modules inspected: {sum(module_types.values())}")
    print(f"\n   Module type counts:")
    for mod_type, count in sorted(module_types.items(), key=lambda x: -x[1])[:20]:
        print(f"     {mod_type}: {count}")
    
    print(f"\n   Quantized modules found: {len(quantized_modules)}")
    if len(quantized_modules) > 0:
        print("   Sample quantized modules:")
        for name, mod_type in quantized_modules[:5]:
            print(f"     {name}: {mod_type}")
    
    print(f"\n   Modules with _packed_params: {len(has_packed_params)}")
    if len(has_packed_params) > 0:
        print("   Sample modules with _packed_params:")
        for name, mod_type in has_packed_params[:5]:
            print(f"     {name}: {mod_type}")
    
    print(f"\n   FP32 Conv2d modules: {len(fp32_modules)}")
    if len(fp32_modules) > 0:
        print("   Sample FP32 Conv2d modules:")
        for name, mod_type in fp32_modules[:5]:
            print(f"     {name}: {mod_type}")
            
except AttributeError as e:
    print(f"   ✗ Failed to iterate modules: {e}")
    print("   This indicates the model has quantized modules that don't support named_modules()")
    print("   This is expected for quantized models with QuantizedConv2d")

# ============================================================================
# Step 4: Test operations that might fail
# ============================================================================
print("\n[Step 4] Testing operations on quantized model...")

# Test 1: named_modules()
print("   [Test 1] Testing named_modules()...")
try:
    modules_list = list(model.model.named_modules())
    print(f"   ✓ named_modules() succeeded: {len(modules_list)} modules")
except (AttributeError, RuntimeError) as e:
    print(f"   ✗ named_modules() failed: {type(e).__name__}: {e}")

# Test 2: .eval()
print("   [Test 2] Testing .eval()...")
try:
    model.model.eval()
    print("   ✓ .eval() succeeded")
except (AttributeError, RuntimeError) as e:
    print(f"   ✗ .eval() failed: {type(e).__name__}: {e}")
    print("   This is the main issue - quantized modules don't support .eval()")

# Test 3: .cpu()
print("   [Test 3] Testing .cpu()...")
try:
    model.model.cpu()
    print("   ✓ .cpu() succeeded")
except (AttributeError, RuntimeError) as e:
    print(f"   ✗ .cpu() failed: {type(e).__name__}: {e}")

# Test 4: .train()
print("   [Test 4] Testing .train()...")
try:
    model.model.train()
    print("   ✓ .train() succeeded")
except (AttributeError, RuntimeError) as e:
    print(f"   ✗ .train() failed: {type(e).__name__}: {e}")

# Test 5: parameters()
print("   [Test 5] Testing .parameters()...")
try:
    params = list(model.model.parameters())
    print(f"   ✓ .parameters() succeeded: {len(params)} parameters")
except (AttributeError, RuntimeError) as e:
    print(f"   ✗ .parameters() failed: {type(e).__name__}: {e}")

# Test 6: state_dict()
print("   [Test 6] Testing .state_dict()...")
try:
    state = model.model.state_dict()
    print(f"   ✓ .state_dict() succeeded: {len(state)} keys")
except (AttributeError, RuntimeError) as e:
    print(f"   ✗ .state_dict() failed: {type(e).__name__}: {e}")

# ============================================================================
# Step 5: Check for FakeQuantize modules (should be none after conversion)
# ============================================================================
print("\n[Step 5] Checking for FakeQuantize modules (should be 0 after conversion)...")
fakequant_count = 0
try:
    for name, module in model.model.named_modules():
        if isinstance(module, FakeQuantize):
            fakequant_count += 1
            if fakequant_count <= 5:
                print(f"   Found FakeQuantize at: {name}")
except (AttributeError, RuntimeError):
    pass

if fakequant_count == 0:
    print("   ✓ No FakeQuantize modules (model is fully converted to INT8)")
else:
    print(f"   ⚠️  Found {fakequant_count} FakeQuantize modules (model may not be fully converted)")

# ============================================================================
# Step 6: Test forward pass
# ============================================================================
print("\n[Step 6] Testing forward pass...")
try:
    model.model.eval()  # Try to set eval mode first
except:
    pass  # Ignore if it fails - model might already be in eval mode

try:
    test_input = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        output = model.model(test_input)
    print("   ✓ Forward pass succeeded")
    if hasattr(output, 'shape'):
        print(f"   Output shape: {output.shape}")
    elif isinstance(output, (list, tuple)):
        print(f"   Output type: {type(output).__name__} with {len(output)} elements")
except Exception as e:
    print(f"   ✗ Forward pass failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Quantized modules: {len(quantized_modules)}")
print(f"FP32 Conv2d modules: {len(fp32_modules)}")
print(f"FakeQuantize modules: {fakequant_count}")
print("\nRecommendations:")
if len(quantized_modules) == 0:
    print("  ⚠️  No quantized modules found - conversion may not have worked")
else:
    print("  ✓ Quantized modules detected")
print("  - Skip .eval() call for quantized models (they're already in eval mode)")
print("  - Wrap .eval() in try-except to handle AttributeError gracefully")
print("  - Quantized models are CPU-only and don't need .cpu() call")
print("=" * 80)

