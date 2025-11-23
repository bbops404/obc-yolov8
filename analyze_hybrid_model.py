#!/usr/bin/env python3
"""Analyze hybrid QAT model to identify FP32, QAT, and PTQ components."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.backends import quantized as torch_quantized_backends

# Ensure the local ultralytics package is importable
REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if not ULTRALYTICS_PATH.exists():
    ULTRALYTICS_PATH = REPO_ROOT / "ultralytics10.24"
if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

# Try to import ultralytics components, but don't fail if they're not available
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    from ultralytics.nn import tasks as ultralytics_tasks
    ensure_module_bookkeeping = getattr(
        ultralytics_tasks,
        "ensure_module_bookkeeping",
        lambda *args, **kwargs: None,
    )
except ImportError:
    # Fallback if ultralytics not available
    ensure_module_bookkeeping = lambda *args, **kwargs: None

# QAT module patterns (from train_hybrid_qat.py)
QAT_MODULE_PATTERNS = ['model.10', 'model.20', 'model.24']  # BoTNet + CoordAtt are QAT


def analyze_hybrid_model(model_path: str | Path):
    """Analyze hybrid model to identify FP32, QAT, and PTQ components."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print("=" * 100)
    print("HYBRID MODEL ANALYSIS")
    print("=" * 100)
    print(f"Model: {model_path}")
    print("=" * 100)
    
    # Set backend
    backend = "qnnpack"
    if backend in torch_quantized_backends.supported_engines:
        torch_quantized_backends.engine = backend
        print(f"Set quantization backend: {backend}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Check backend
    checkpoint_backend = checkpoint.get('backend', None)
    if checkpoint_backend:
        backend = checkpoint_backend
        torch_quantized_backends.engine = backend
        print(f"Using checkpoint backend: {backend}")
    
    # Check checkpoint metadata for QAT information
    print("\n" + "=" * 100)
    print("CHECKPOINT METADATA")
    print("=" * 100)
    checkpoint_keys = [k for k in checkpoint.keys() if not k.startswith('model')]
    print(f"Checkpoint keys (non-model): {checkpoint_keys}")
    
    # Check for sensitive_modules or qat_modules in checkpoint
    sensitive_modules_from_checkpoint = checkpoint.get('sensitive_modules', None)
    qat_modules_from_checkpoint = checkpoint.get('qat_modules', None)
    hybrid_qat = checkpoint.get('hybrid_qat', False)
    qat_layers_from_checkpoint = checkpoint.get('qat_layers', None)
    ptq_layers_from_checkpoint = checkpoint.get('ptq_layers', None)
    
    # Determine QAT patterns - use checkpoint metadata if available, otherwise use default
    qat_patterns = QAT_MODULE_PATTERNS.copy()  # Start with default
    if sensitive_modules_from_checkpoint is not None:
        print(f"✓ Found sensitive_modules in checkpoint: {sensitive_modules_from_checkpoint}")
        qat_patterns = sensitive_modules_from_checkpoint
    elif qat_modules_from_checkpoint is not None:
        print(f"✓ Found qat_modules in checkpoint: {qat_modules_from_checkpoint}")
        qat_patterns = qat_modules_from_checkpoint
    else:
        print("⚠ No sensitive_modules or qat_modules found in checkpoint metadata")
        print(f"  Using default QAT patterns: {qat_patterns}")
    
    print(f"  → Will use QAT patterns: {qat_patterns}")
    
    if hybrid_qat:
        print(f"✓ Model is hybrid QAT: {hybrid_qat}")
    
    # Show QAT and PTQ layer lists if available
    if qat_layers_from_checkpoint is not None:
        print(f"\n✓ QAT Layers (from checkpoint): {len(qat_layers_from_checkpoint)} layers")
        if len(qat_layers_from_checkpoint) <= 20:
            for layer in qat_layers_from_checkpoint:
                print(f"    - {layer}")
        else:
            print(f"    (Showing first 10 and last 10 of {len(qat_layers_from_checkpoint)} layers)")
            for layer in qat_layers_from_checkpoint[:10]:
                print(f"    - {layer}")
            print(f"    ... ({len(qat_layers_from_checkpoint) - 20} more layers) ...")
            for layer in qat_layers_from_checkpoint[-10:]:
                print(f"    - {layer}")
    
    if ptq_layers_from_checkpoint is not None:
        print(f"\n✓ PTQ Layers (from checkpoint): {len(ptq_layers_from_checkpoint)} layers")
        if len(ptq_layers_from_checkpoint) <= 20:
            for layer in ptq_layers_from_checkpoint:
                print(f"    - {layer}")
        else:
            print(f"    (Showing first 10 and last 10 of {len(ptq_layers_from_checkpoint)} layers)")
            for layer in ptq_layers_from_checkpoint[:10]:
                print(f"    - {layer}")
            print(f"    ... ({len(ptq_layers_from_checkpoint) - 20} more layers) ...")
            for layer in ptq_layers_from_checkpoint[-10:]:
                print(f"    - {layer}")
    
    print("=" * 100)
    
    # Load model
    if 'model' in checkpoint:
        model = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        raise ValueError("Cannot analyze from state_dict alone - need full model")
    else:
        raise ValueError("Checkpoint must contain 'model' or 'model_state_dict'")
    
    # Ensure bookkeeping
    ensure_module_bookkeeping(model, recursive=True)
    
    # Import module types
    import torch.nn as nn
    from ultralytics.nn.ODConv import ODConv
    from ultralytics.nn.BoTNet import BoTNet
    from ultralytics.nn.CA_Attention import CoordAtt
    
    # Categorize modules
    fp32_modules = []
    qat_modules = []  # Modules that were QAT (now quantized but from QAT training)
    ptq_modules = []  # Modules that were PTQ (quantized via PTQ)
    quantized_modules = []  # All quantized modules (INT8)
    
    # Statistics
    fp32_params = 0
    qat_params = 0
    ptq_params = 0
    quantized_params = 0
    
    print("\nAnalyzing model structure...")
    
    # Debug: Check model.10 and related modules
    print("\n" + "=" * 100)
    print("INSPECTING MODEL.10 (BoTNet)")
    print("=" * 100)
    
    model_10_found = False
    for name, module in model.named_modules():
        if name == "model.10" or name.startswith("model.10."):
            if not model_10_found:
                print(f"\n🔍 Found model.10:")
                print(f"  Full name: {name}")
                print(f"  Module type: {type(module).__name__}")
                print(f"  Module path: {type(module).__module__}")
                print(f"  Is BoTNet instance: {isinstance(module, BoTNet)}")
                print(f"  Has _packed_params: {hasattr(module, '_packed_params')}")
                print(f"  Regular params count: {sum(p.numel() for p in module.parameters())}")
                model_10_found = True
                break
    
    # Check model.10 submodules
    print(f"\n🔍 Inspecting model.10 submodules:")
    model_10_modules = []
    for name, module in model.named_modules():
        if name.startswith("model.10."):
            module_type = type(module).__name__
            module_path = type(module).__module__
            is_quantized = 'quantized' in module_path.lower() or (hasattr(module, '_packed_params') and module._packed_params is not None)
            model_10_modules.append((name, module_type, module_path, is_quantized))
    
    # Show first 10 submodules
    for name, mod_type, mod_path, is_quant in model_10_modules[:10]:
        print(f"  {name:<50} {mod_type:<20} quantized={is_quant}")
    if len(model_10_modules) > 10:
        print(f"  ... and {len(model_10_modules) - 10} more submodules")
    
    print("=" * 100)
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_path = type(module).__module__
        
        # Check if this module is in QAT patterns (BoTNet or CoordAtt that were QAT trained)
        is_qat_pattern = any(pattern in name for pattern in qat_patterns)
        
        # Check if it's a special FP32 module (ODConv is always FP32, but BoTNet/CoordAtt can be QAT)
        # Only treat as FP32 if it's ODConv, or if it's BoTNet/CoordAtt but NOT in QAT patterns
        is_fp32_special = isinstance(module, ODConv) or (
            isinstance(module, (BoTNet, CoordAtt)) and not is_qat_pattern
        )
        
        # Check if it's quantized
        is_quantized = False
        quantized_weight_params = 0
        
        # Check for _packed_params (sign of quantization)
        has_packed_params = hasattr(module, '_packed_params') and module._packed_params is not None
        
        if module_type == 'Conv2d' and ('quantized' in module_path.lower() or has_packed_params):
            is_quantized = True
            if has_packed_params:
                try:
                    packed_params = module._packed_params
                    # _packed_params for Conv2d can be a tuple or have weight() method
                    if hasattr(packed_params, 'weight'):
                        weight = packed_params.weight()
                        quantized_weight_params = weight.numel()
                    elif isinstance(packed_params, tuple) and len(packed_params) > 0:
                        # First element is usually the quantized weight
                        weight = packed_params[0]
                        if hasattr(weight, 'numel'):
                            quantized_weight_params = weight.numel()
                except Exception as e:
                    pass
        elif hasattr(module, 'conv') and isinstance(module.conv, nn.Module):
            conv_module = module.conv
            conv_path = type(conv_module).__module__
            conv_has_packed = hasattr(conv_module, '_packed_params') and conv_module._packed_params is not None
            if 'quantized' in conv_path.lower() or conv_has_packed:
                is_quantized = True
                if conv_has_packed:
                    try:
                        packed_params = conv_module._packed_params
                        if hasattr(packed_params, 'weight'):
                            weight = packed_params.weight()
                            quantized_weight_params = weight.numel()
                        elif isinstance(packed_params, tuple) and len(packed_params) > 0:
                            weight = packed_params[0]
                            if hasattr(weight, 'numel'):
                                quantized_weight_params = weight.numel()
                    except:
                        pass
        elif module_type == 'Linear':
            # Check if it's quantized - either by path or by having _packed_params
            if 'quantized' in module_path.lower() or has_packed_params:
                is_quantized = True
                if has_packed_params:
                    try:
                        packed_params = module._packed_params
                        # For Linear, _packed_params typically has weight() method
                        if hasattr(packed_params, 'weight'):
                            weight = packed_params.weight()
                            quantized_weight_params = weight.numel()
                        elif isinstance(packed_params, tuple) and len(packed_params) > 0:
                            # Could be tuple format
                            weight = packed_params[0]
                            if hasattr(weight, 'numel'):
                                quantized_weight_params = weight.numel()
                        # Also try direct access if it's a tensor
                        elif hasattr(packed_params, 'numel'):
                            quantized_weight_params = packed_params.numel()
                    except Exception as e:
                        pass
        
        # Count parameters (for FP32 modules or fallback)
        module_params = sum(p.numel() for p in module.parameters())
        
        # For quantized modules, if we couldn't get weight params, try to estimate from module attributes
        if is_quantized and quantized_weight_params == 0:
            # Try to get shape info from module attributes
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                # Linear layer
                quantized_weight_params = module.in_features * module.out_features
                if hasattr(module, 'bias') and module.bias is not None:
                    quantized_weight_params += module.out_features  # bias
            elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                # Conv2d layer
                kernel_size = getattr(module, 'kernel_size', (1, 1))
                if isinstance(kernel_size, (tuple, list)):
                    k = kernel_size[0] * kernel_size[1] if len(kernel_size) >= 2 else kernel_size[0]
                else:
                    k = kernel_size * kernel_size
                quantized_weight_params = module.in_channels * module.out_channels * k
                if hasattr(module, 'bias') and module.bias is not None:
                    quantized_weight_params += module.out_channels  # bias
        
        # Categorize
        if is_fp32_special:
            fp32_modules.append((name, module_type, module_params))
            fp32_params += module_params
        elif is_quantized:
            quantized_modules.append((name, module_type, quantized_weight_params if quantized_weight_params > 0 else module_params))
            quantized_params += (quantized_weight_params if quantized_weight_params > 0 else module_params)
            
            # Determine if it was QAT or PTQ based on module name patterns
            # Check if this module name matches any QAT pattern
            is_qat = any(pattern in name for pattern in qat_patterns)
            if is_qat:
                qat_modules.append((name, module_type, quantized_weight_params if quantized_weight_params > 0 else module_params))
                qat_params += (quantized_weight_params if quantized_weight_params > 0 else module_params)
            else:
                ptq_modules.append((name, module_type, quantized_weight_params if quantized_weight_params > 0 else module_params))
                ptq_params += (quantized_weight_params if quantized_weight_params > 0 else module_params)
    
    # Print results
    print("\n" + "=" * 100)
    print("MODULE BREAKDOWN")
    print("=" * 100)
    
    total_params = fp32_params + quantized_params
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  FP32 Parameters:      {fp32_params:,} ({100*fp32_params/total_params:.2f}%)")
    print(f"  Quantized Parameters: {quantized_params:,} ({100*quantized_params/total_params:.2f}%)")
    print(f"    - QAT (trained):    {qat_params:,} ({100*qat_params/total_params:.2f}%)")
    print(f"    - PTQ (calibrated): {ptq_params:,} ({100*ptq_params/total_params:.2f}%)")
    
    print(f"\n📦 MODULE COUNTS:")
    print(f"  FP32 Modules:          {len(fp32_modules)}")
    print(f"  QAT Modules (INT8):   {len(qat_modules)}")
    print(f"  PTQ Modules (INT8):   {len(ptq_modules)}")
    print(f"  Total Quantized:      {len(quantized_modules)}")
    
    # Print FP32 modules
    print(f"\n{'='*100}")
    print("FP32 MODULES (Kept in Full Precision)")
    print(f"{'='*100}")
    if fp32_modules:
        print(f"\n{len(fp32_modules)} modules kept in FP32:")
        for name, mod_type, params in sorted(fp32_modules, key=lambda x: -x[2]):
            print(f"  {name:<60} {mod_type:<20} {params:>12,} params")
    else:
        print("  No FP32 modules found")
    
    # Print QAT modules
    print(f"\n{'='*100}")
    print("QAT MODULES (Quantization-Aware Training - Now INT8)")
    print(f"{'='*100}")
    if qat_modules:
        print(f"\n{len(qat_modules)} modules trained with QAT (now quantized to INT8):")
        for name, mod_type, params in sorted(qat_modules, key=lambda x: -x[2]):
            print(f"  {name:<60} {mod_type:<20} {params:>12,} params")
    else:
        print("  No QAT modules found")
    
    # Print PTQ modules
    print(f"\n{'='*100}")
    print("PTQ MODULES (Post-Training Quantization - INT8)")
    print(f"{'='*100}")
    if ptq_modules:
        print(f"\n{len(ptq_modules)} modules quantized with PTQ (INT8):")
        # Show first 20 and last 20
        if len(ptq_modules) > 40:
            print("  (Showing first 20 and last 20 modules)")
            for name, mod_type, params in sorted(ptq_modules, key=lambda x: -x[2])[:20]:
                print(f"  {name:<60} {mod_type:<20} {params:>12,} params")
            print("  ...")
            for name, mod_type, params in sorted(ptq_modules, key=lambda x: -x[2])[-20:]:
                print(f"  {name:<60} {mod_type:<20} {params:>12,} params")
        else:
            for name, mod_type, params in sorted(ptq_modules, key=lambda x: -x[2]):
                print(f"  {name:<60} {mod_type:<20} {params:>12,} params")
    else:
        print("  No PTQ modules found")
    
    # Calculate sizes
    fp32_size_mb = (fp32_params * 4) / (1024 * 1024)  # FP32 = 4 bytes
    qat_size_mb = (qat_params * 1) / (1024 * 1024)  # INT8 = 1 byte
    ptq_size_mb = (ptq_params * 1) / (1024 * 1024)  # INT8 = 1 byte
    total_size_mb = fp32_size_mb + qat_size_mb + ptq_size_mb
    
    print(f"\n{'='*100}")
    print("SIZE BREAKDOWN")
    print(f"{'='*100}")
    print(f"  FP32 Size:            {fp32_size_mb:.2f} MB ({100*fp32_size_mb/total_size_mb:.2f}%)")
    print(f"  QAT Size (INT8):      {qat_size_mb:.2f} MB ({100*qat_size_mb/total_size_mb:.2f}%)")
    print(f"  PTQ Size (INT8):      {ptq_size_mb:.2f} MB ({100*ptq_size_mb/total_size_mb:.2f}%)")
    print(f"  Total Estimated:      {total_size_mb:.2f} MB")
    
    print("\n" + "=" * 100)
    
    return {
        'fp32_modules': fp32_modules,
        'qat_modules': qat_modules,
        'ptq_modules': ptq_modules,
        'fp32_params': fp32_params,
        'qat_params': qat_params,
        'ptq_params': ptq_params,
        'fp32_size_mb': fp32_size_mb,
        'qat_size_mb': qat_size_mb,
        'ptq_size_mb': ptq_size_mb,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze hybrid QAT model structure")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ubuntu/obc-yolov8/obc-yolov8/runs/detect/train_hybrid_qat/weights/last_int8.pt",
        help="Path to hybrid INT8 model"
    )
    
    args = parser.parse_args()
    
    try:
        analyze_hybrid_model(args.model)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

