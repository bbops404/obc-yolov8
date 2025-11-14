#!/usr/bin/env python3
"""Inspect what is quantized in the INT8 model."""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

import torch
from ultralytics.nn.tasks import ensure_module_bookkeeping
from ultralytics.utils import LOGGER

def inspect_quantization(model, prefix=""):
    """Recursively inspect model to find quantized operations."""
    quantized_ops = []
    fp32_ops = []
    fakequant_ops = []
    other_ops = []
    
    for name, module in model.named_modules():
        full_name = f"{prefix}.{name}" if prefix else name
        module_type = type(module).__name__
        module_path = type(module).__module__
        
        # Check for quantized operations by module type or path
        is_quantized_module = (
            'Quantized' in module_type or 
            'quantized' in module_path.lower()
        )
        
        # Check for FakeQuantize (shouldn't be in INT8, but check anyway)
        if 'FakeQuantize' in module_type:
            fakequant_ops.append((full_name, module_type, module))
        # Check for quantized Conv2d/Linear directly (not wrapped)
        # Skip Conv2d layers that are inside Conv wrappers (they'll be handled by the Conv wrapper check)
        # Quantized Conv2d modules are from torch.ao.nn.quantized, not torch.nn
        elif (not full_name.endswith('.conv') and  # Skip .conv children (handled by Conv wrapper check)
              (isinstance(module, torch.nn.Conv2d) or 
               isinstance(module, torch.nn.Linear) or
               (module_type == 'Conv2d' and 'quantized' in module_path.lower()) or
               (module_type == 'Linear' and 'quantized' in module_path.lower()))):
            # Check if it's quantized by looking for _packed_params or quantized module path
            is_quantized = (
                hasattr(module, '_packed_params') or 
                is_quantized_module
            )
            
            if is_quantized:
                quantized_ops.append((full_name, f"{module_type} (Quantized)", module))
            else:
                fp32_ops.append((full_name, module_type, module))
        # Check for BatchNorm2d (usually not quantized directly)
        elif isinstance(module, torch.nn.BatchNorm2d):
            fp32_ops.append((full_name, module_type, module))
        # Check inside Conv wrapper modules (they contain self.conv which might be quantized)
        elif hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Module):
            conv_module = module.conv
            conv_type = type(conv_module).__name__
            conv_module_path = type(conv_module).__module__
            
            # Check if it's quantized by looking for _packed_params or quantized module path
            is_quantized = (
                hasattr(conv_module, '_packed_params') or 
                'quantized' in conv_module_path.lower() or
                'Quantized' in conv_type
            )
            
            if is_quantized:
                quantized_ops.append((f"{full_name}.conv", f"{conv_type} (Quantized, inside {module_type})", conv_module))
            elif isinstance(conv_module, torch.nn.Conv2d):
                fp32_ops.append((f"{full_name}.conv", f"Conv2d (FP32, inside {module_type})", conv_module))
        else:
            # Other operations
            if not any(x in module_type for x in ['Sequential', 'ModuleList', 'ModuleDict', 'Container']):
                other_ops.append((full_name, module_type, module))
    
    return quantized_ops, fp32_ops, fakequant_ops, other_ops

def get_module_info(module):
    """Get information about a module."""
    info = []
    
    # Show module path
    module_path = type(module).__module__
    if 'quantized' in module_path.lower():
        info.append(f"module={module_path}")
    
    # Check for weight shape
    if hasattr(module, 'weight') and module.weight is not None:
        if isinstance(module.weight, torch.Tensor):
            info.append(f"weight_shape={tuple(module.weight.shape)}")
        elif hasattr(module.weight, 'shape'):
            info.append(f"weight_shape={tuple(module.weight.shape)}")
    
    # Check for bias
    if hasattr(module, 'bias') and module.bias is not None:
        info.append("has_bias=True")
    elif hasattr(module, 'bias'):
        info.append("has_bias=False")
    
    # Check for quantization parameters
    if hasattr(module, 'scale'):
        if isinstance(module.scale, torch.Tensor):
            info.append(f"scale={module.scale.item():.6f}")
        else:
            info.append(f"scale={module.scale}")
    
    if hasattr(module, 'zero_point'):
        if isinstance(module.zero_point, torch.Tensor):
            info.append(f"zero_point={module.zero_point.item()}")
        else:
            info.append(f"zero_point={module.zero_point}")
    
    # Check for _packed_params
    if hasattr(module, '_packed_params'):
        info.append("_packed_params=True")
        # Try to get scale from packed params if available
        try:
            packed = module._packed_params
            if hasattr(packed, 'scale'):
                info.append(f"packed_scale={packed.scale}")
        except:
            pass
    
    return ", ".join(info) if info else ""

def main():
    parser = argparse.ArgumentParser(description="Inspect quantized layers in an INT8 model checkpoint")
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        default="runs/detect/train_qat14/weights/last_int8.pt",
        help="Path to INT8 checkpoint file (default: runs/detect/train_qat14/weights/last_int8.pt)"
    )
    args = parser.parse_args()
    
    int8_path = args.checkpoint
    
    LOGGER.info("="*80)
    LOGGER.info("Inspecting INT8 Model Quantization")
    LOGGER.info("="*80)
    LOGGER.info(f"Loading INT8 checkpoint: {int8_path}")
    
    try:
        checkpoint = torch.load(int8_path, map_location='cpu', weights_only=False)
        model = checkpoint.get('model')
        
        if model is None:
            LOGGER.error("No model found in checkpoint")
            return
        
        # Ensure bookkeeping for proper traversal
        ensure_module_bookkeeping(model, recursive=True)
        model.eval()
        
        LOGGER.info("Analyzing quantized operations...\n")
        
        quantized_ops, fp32_ops, fakequant_ops, other_ops = inspect_quantization(model)
        
        # Group quantized ops by type
        quantized_by_type = defaultdict(list)
        for name, op_type, module in quantized_ops:
            quantized_by_type[op_type].append((name, module))
        
        # Group FP32 ops by type
        fp32_by_type = defaultdict(list)
        for name, op_type, module in fp32_ops:
            fp32_by_type[op_type].append((name, module))
        
        LOGGER.info("="*80)
        LOGGER.info("QUANTIZED OPERATIONS")
        LOGGER.info("="*80)
        
        total_quantized = 0
        for op_type in sorted(quantized_by_type.keys()):
            ops = quantized_by_type[op_type]
            total_quantized += len(ops)
            LOGGER.info(f"\n{op_type}: {len(ops)} operations")
            LOGGER.info("-" * 80)
            
            # Show all examples
            for i, (name, module) in enumerate(ops):
                info = get_module_info(module)
                LOGGER.info(f"  [{i+1}] {name}")
                if info:
                    LOGGER.info(f"       {info}")
        
        LOGGER.info(f"\nTotal quantized operations: {total_quantized}")
        
        LOGGER.info("\n" + "="*80)
        LOGGER.info("FP32 OPERATIONS (Not Quantized)")
        LOGGER.info("="*80)
        
        total_fp32 = 0
        for op_type in sorted(fp32_by_type.keys()):
            ops = fp32_by_type[op_type]
            total_fp32 += len(ops)
            LOGGER.info(f"\n{op_type}: {len(ops)} operations")
            LOGGER.info("-" * 80)
            
            # Show all examples
            for i, (name, module) in enumerate(ops):
                info = get_module_info(module)
                LOGGER.info(f"  [{i+1}] {name}")
                if info:
                    LOGGER.info(f"       {info}")
        
        LOGGER.info(f"\nTotal FP32 operations: {total_fp32}")
        
        if fakequant_ops:
            LOGGER.info("\n" + "="*80)
            LOGGER.info("WARNING: FakeQuantize operations found (should not be in INT8)")
            LOGGER.info("="*80)
            for name, op_type, module in fakequant_ops:
                LOGGER.info(f"  {name}: {op_type}")
        
        # Summary
        LOGGER.info("\n" + "="*80)
        LOGGER.info("QUANTIZATION SUMMARY")
        LOGGER.info("="*80)
        LOGGER.info(f"Quantized operations: {total_quantized}")
        LOGGER.info(f"FP32 operations: {total_fp32}")
        LOGGER.info(f"Quantization ratio: {total_quantized/(total_quantized+total_fp32)*100:.1f}%")
        
        # Detailed breakdown
        LOGGER.info("\n" + "="*80)
        LOGGER.info("DETAILED BREAKDOWN BY TYPE")
        LOGGER.info("="*80)
        
        LOGGER.info("\nQuantized:")
        for op_type in sorted(quantized_by_type.keys()):
            count = len(quantized_by_type[op_type])
            LOGGER.info(f"  {op_type}: {count}")
        
        LOGGER.info("\nFP32:")
        for op_type in sorted(fp32_by_type.keys()):
            count = len(fp32_by_type[op_type])
            LOGGER.info(f"  {op_type}: {count}")
        
    except Exception as e:
        LOGGER.error(f"Error inspecting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

