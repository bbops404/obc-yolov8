#!/usr/bin/env python3
"""
Simple script to identify PTQ vs QAT layers based on module names.

Usage:
    python identify_layers_simple.py <path_to_int8_model.pt> [--sensitive-modules model.10]
"""

import torch
import sys
from pathlib import Path
from collections import defaultdict

# Add ultralytics path to sys.path if it exists
ultralytics_path = Path(__file__).parent / "obc-yolov8" / "ultralytics10.24"
if ultralytics_path.exists():
    sys.path.insert(0, str(ultralytics_path.parent))
    sys.path.insert(0, str(ultralytics_path))

def identify_layers(checkpoint_path, sensitive_modules=None):
    """Identify QAT vs PTQ layers from checkpoint state_dict."""
    
    if sensitive_modules is None:
        sensitive_modules = ['model.10']  # Default: BoTNet
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("=" * 80)
    print(f"Identifying PTQ vs QAT Layers: {checkpoint_path.name}")
    print("=" * 80)
    print(f"Sensitive modules (QAT): {', '.join(sensitive_modules)}")
    print()
    
    # Try to load just the state_dict using pickle with custom class loader
    import pickle
    import io
    
    state_dict = None
    checkpoint = None
    
    # Custom unpickler that skips unknown classes
    class SkipUnknownUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # For known torch/collections types, use normal loading
            if module == 'torch._utils' or module.startswith('torch.'):
                try:
                    return super().find_class(module, name)
                except:
                    pass
            if module == 'collections' or module == 'collections.abc':
                try:
                    return super().find_class(module, name)
                except:
                    pass
            # For unknown classes (like ultralytics modules), return a dummy
            class Dummy:
                def __init__(self, *args, **kwargs):
                    pass
            return Dummy
    
    try:
        with open(checkpoint_path, 'rb') as f:
            unpickler = SkipUnknownUnpickler(f)
            checkpoint = unpickler.load()
            
        if isinstance(checkpoint, dict):
            # Try to get state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Check if it's a state_dict itself (keys look like layer names)
                sample_keys = list(checkpoint.keys())[:5]
                if sample_keys and any('.' in str(k) for k in sample_keys):
                    state_dict = checkpoint
    except Exception as e:
        print(f"⚠️  Could not load with custom unpickler: {e}")
        print("   Trying standard torch.load...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', None))
        except Exception as e2:
            print(f"❌ Could not load checkpoint: {e2}")
            print("\n💡 Tip: The checkpoint needs to be re-saved with metadata.")
            print("   For now, identify layers manually:")
            print(f"   - QAT layers: modules containing '{sensitive_modules[0]}'")
            print("   - PTQ layers: all other quantized modules")
            return
    
    if state_dict is None:
        print("❌ Could not extract state_dict from checkpoint")
        return
    
    # Analyze layer names
    qat_layers = []
    ptq_layers = []
    fp32_layers = []
    
    # Get all layer names from state_dict
    all_layers = set()
    for key in state_dict.keys():
        # Extract module name (e.g., 'model.10.cv1.conv.weight' -> 'model.10.cv1.conv')
        parts = key.split('.')
        # Remove weight/bias/etc suffixes
        if parts[-1] in ['weight', 'bias', '_packed_params', 'scale', 'zero_point']:
            layer_name = '.'.join(parts[:-1])
        else:
            layer_name = '.'.join(parts)
        all_layers.add(layer_name)
    
    # Categorize layers
    for layer_name in sorted(all_layers):
        # Check if it matches sensitive_modules pattern
        is_sensitive = any(pattern in layer_name for pattern in sensitive_modules)
        
        # Check if it's quantized (has _packed_params or quantized weight)
        is_quantized = False
        for key in state_dict.keys():
            if layer_name in key:
                if '_packed_params' in key or 'weight' in key:
                    # Check if weight is quantized (has scale/zero_point or is quantized tensor)
                    weight_key = f"{layer_name}.weight"
                    scale_key = f"{layer_name}.scale"
                    zp_key = f"{layer_name}.zero_point"
                    if weight_key in state_dict or scale_key in state_dict or zp_key in state_dict:
                        is_quantized = True
                    elif '_packed_params' in key:
                        is_quantized = True
                break
        
        if is_quantized:
            if is_sensitive:
                qat_layers.append(layer_name)
            else:
                ptq_layers.append(layer_name)
        else:
            fp32_layers.append(layer_name)
    
    # Print results
    print("📊 Summary:")
    print(f"  QAT layers (sensitive, fine-tuned):     {len(qat_layers)}")
    print(f"  PTQ layers (calibrated only):          {len(ptq_layers)}")
    print(f"  FP32 layers (not quantized):            {len(fp32_layers)}")
    print()
    
    print("🔧 QAT Layers (from sensitive modules):")
    print("-" * 80)
    if qat_layers:
        for layer in qat_layers[:30]:
            print(f"  ✓ {layer}")
        if len(qat_layers) > 30:
            print(f"  ... and {len(qat_layers) - 30} more")
    else:
        print("  (none found - check sensitive_modules patterns)")
    print()
    
    print("📐 PTQ Layers (calibrated, not fine-tuned):")
    print("-" * 80)
    if ptq_layers:
        for layer in ptq_layers[:30]:
            print(f"  • {layer}")
        if len(ptq_layers) > 30:
            print(f"  ... and {len(ptq_layers) - 30} more")
    else:
        print("  (none found)")
    print()
    
    # Group by module pattern
    print("📦 Grouped by Module Pattern:")
    print("-" * 80)
    qat_by_pattern = defaultdict(list)
    ptq_by_pattern = defaultdict(list)
    
    for layer in qat_layers:
        parts = layer.split('.')
        if len(parts) >= 2:
            pattern = '.'.join(parts[:2])
            qat_by_pattern[pattern].append(layer)
    
    for layer in ptq_layers:
        parts = layer.split('.')
        if len(parts) >= 2:
            pattern = '.'.join(parts[:2])
            ptq_by_pattern[pattern].append(layer)
    
    for pattern in sorted(set(list(qat_by_pattern.keys()) + list(ptq_by_pattern.keys()))):
        qat_count = len(qat_by_pattern.get(pattern, []))
        ptq_count = len(ptq_by_pattern.get(pattern, []))
        is_sensitive = any(p in pattern for p in sensitive_modules)
        status = "🔧 QAT" if is_sensitive else "📐 PTQ"
        print(f"  {pattern}: {status} ({qat_count} QAT, {ptq_count} PTQ)")
    
    return {
        'qat_layers': qat_layers,
        'ptq_layers': ptq_layers,
        'fp32_layers': fp32_layers
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python identify_layers_simple.py <path_to_int8_model.pt> [--sensitive-modules model.10,model.19]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    sensitive_modules = None
    
    if '--sensitive-modules' in sys.argv:
        idx = sys.argv.index('--sensitive-modules')
        if idx + 1 < len(sys.argv):
            sensitive_modules = sys.argv[idx + 1].split(',')
    
    try:
        identify_layers(checkpoint_path, sensitive_modules)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

