"""
Debug script to find which quantized Conv2d modules are causing the _backward_hooks error
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'obc-yolov8' / 'ultralytics10.24'))

import torch
import torch.nn as nn

def main():
    int8_checkpoint = 'runs/detect/train/weights/best_int8.pt'
    
    print("=" * 80)
    print("Debugging Quantized Conv2d Modules")
    print("=" * 80)
    
    # Load INT8 model
    print(f"\n[1/2] Loading INT8 model from {int8_checkpoint}...")
    checkpoint = torch.load(int8_checkpoint, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        model = checkpoint['model']
        backend = checkpoint.get('backend', 'fbgemm')
    else:
        print("Error: INT8 checkpoint doesn't contain model object")
        return
    
    print(f"   Backend: {backend}")
    torch.backends.quantized.engine = backend
    
    # Find all QuantizedConv2d modules
    print(f"\n[2/2] Finding QuantizedConv2d modules...")
    
    quantized_conv2d = []
    problematic_modules = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # Check if it's a QuantizedConv2d
        if 'QuantizedConv2d' in module_type or hasattr(module, '_packed_params'):
            quantized_conv2d.append(name)
            
            # Check if it's inside a custom module
            parent_path = '.'.join(name.split('.')[:-1])
            if parent_path:
                try:
                    # Check all parent levels
                    path_parts = name.split('.')
                    for i in range(len(path_parts) - 1, 0, -1):
                        check_path = '.'.join(path_parts[:i])
                        try:
                            parent = dict(model.named_modules())[check_path]
                            parent_type = type(parent).__name__
                            
                            # Check if parent is a custom module
                            if parent_type in ['Conv', 'C2f', 'C2', 'C3', 'C3x', 'SPPF', 'SPP', 
                                              'Attention', 'BottleneckTransformer', 'MHSA', 
                                              'CoordAtt', 'CA_Attention', 'Detect', 'DFL']:
                                problematic_modules.append((name, parent_type, check_path))
                                break
                            elif 'ultralytics' in str(type(parent).__module__):
                                problematic_modules.append((name, parent_type, check_path))
                                break
                        except:
                            pass
                except:
                    pass
    
    print(f"\n   Total QuantizedConv2d modules: {len(quantized_conv2d)}")
    print(f"   Problematic QuantizedConv2d (inside custom modules): {len(problematic_modules)}")
    
    if problematic_modules:
        print(f"\n   ⚠️  Found {len(problematic_modules)} QuantizedConv2d modules inside custom wrappers:")
        for name, parent_type, parent_path in problematic_modules[:20]:
            print(f"     - {name}")
            print(f"       Parent: {parent_type} at {parent_path}")
    
    # Also check regular Conv2d that might have been converted
    print(f"\n   Checking for regular Conv2d modules that might be problematic...")
    regular_conv2d_in_custom = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parent_path = '.'.join(name.split('.')[:-1])
            if parent_path:
                try:
                    parent = dict(model.named_modules())[parent_path]
                    parent_type = type(parent).__name__
                    
                    if parent_type in ['Conv', 'C2f', 'C2', 'C3', 'C3x', 'SPPF', 'SPP']:
                        regular_conv2d_in_custom.append((name, parent_type))
                except:
                    pass
    
    if regular_conv2d_in_custom:
        print(f"   ⚠️  Found {len(regular_conv2d_in_custom)} regular Conv2d inside custom modules:")
        for name, parent_type in regular_conv2d_in_custom[:10]:
            print(f"     - {name} (parent: {parent_type})")
    
    # Try to identify which module is causing the error
    print(f"\n   Attempting to identify the problematic module by testing forward pass...")
    try:
        test_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            _ = model(test_input)
        print("   ✓ Forward pass succeeded!")
    except AttributeError as e:
        error_msg = str(e)
        print(f"   ✗ Forward pass failed: {error_msg}")
        
        # Try to identify which module is accessed
        if "'Conv2d' object has no attribute" in error_msg:
            print("\n   This suggests a QuantizedConv2d is being accessed as a regular Conv2d")
            print("   The issue is likely in one of the problematic modules above")
    
    return problematic_modules

if __name__ == '__main__':
    main()

