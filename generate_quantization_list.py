#!/usr/bin/env python3
"""Generate a simple list of quantized and non-quantized operations."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

import torch
from ultralytics.nn.tasks import ensure_module_bookkeeping

def inspect_quantization(model, prefix=""):
    """Recursively inspect model to find quantized operations."""
    quantized_ops = []
    fp32_ops = []
    
    for name, module in model.named_modules():
        full_name = f"{prefix}.{name}" if prefix else name
        module_type = type(module).__name__
        
        # Check for quantized operations
        if 'Quantized' in module_type:
            quantized_ops.append((full_name, module_type))
        # Check for standard operations that might be quantizable
        elif isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            # Check if it has _packed_params (sign of quantization)
            if hasattr(module, '_packed_params'):
                quantized_ops.append((full_name, f"{module_type} (Quantized)"))
            else:
                fp32_ops.append((full_name, module_type))
        # Check inside Conv wrapper modules
        elif hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Module):
            conv_module = module.conv
            conv_type = type(conv_module).__name__
            conv_module_path = type(conv_module).__module__
            
            # Check if it's quantized
            is_quantized = (
                hasattr(conv_module, '_packed_params') or 
                'quantized' in conv_module_path.lower() or
                'Quantized' in conv_type
            )
            
            if is_quantized:
                quantized_ops.append((f"{full_name}.conv", f"Conv2d (Quantized)"))
            elif isinstance(conv_module, torch.nn.Conv2d):
                fp32_ops.append((f"{full_name}.conv", f"Conv2d (FP32)"))
    
    return quantized_ops, fp32_ops

def main():
    int8_path = "runs/detect/train_qat14/weights/last_int8.pt"
    
    print("="*80)
    print("QUANTIZED vs NON-QUANTIZED OPERATIONS LIST")
    print("="*80)
    print(f"Model: {int8_path}\n")
    
    checkpoint = torch.load(int8_path, map_location='cpu', weights_only=False)
    model = checkpoint.get('model')
    
    if model is None:
        print("ERROR: No model found in checkpoint")
        return
    
    ensure_module_bookkeeping(model, recursive=True)
    model.eval()
    
    quantized_ops, fp32_ops = inspect_quantization(model)
    
    # Sort by name
    quantized_ops.sort(key=lambda x: x[0])
    fp32_ops.sort(key=lambda x: x[0])
    
    print("="*80)
    print(f"QUANTIZED OPERATIONS ({len(quantized_ops)} total)")
    print("="*80)
    for i, (name, op_type) in enumerate(quantized_ops, 1):
        print(f"{i:3d}. {name}")
    
    print("\n" + "="*80)
    print(f"NON-QUANTIZED (FP32) OPERATIONS ({len(fp32_ops)} total)")
    print("="*80)
    for i, (name, op_type) in enumerate(fp32_ops, 1):
        print(f"{i:3d}. {name} ({op_type})")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total operations: {len(quantized_ops) + len(fp32_ops)}")
    print(f"Quantized: {len(quantized_ops)} ({len(quantized_ops)/(len(quantized_ops)+len(fp32_ops))*100:.1f}%)")
    print(f"FP32: {len(fp32_ops)} ({len(fp32_ops)/(len(quantized_ops)+len(fp32_ops))*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()

