"""
Debug script to diagnose quantized backend dispatch issues.

This script inspects the INT8 model to identify:
1. Which modules are quantized vs FP32
2. Backend engine configuration
3. Quantization parameters (scale/zero_point)
4. Tensor backend issues in BoTNet/MHSA modules
"""

import sys
from pathlib import Path

# Add ultralytics path to sys.path
REPO_ROOT = Path(__file__).parent.resolve()
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if not ULTRALYTICS_PATH.exists():
    ULTRALYTICS_PATH = REPO_ROOT / "ultralytics10.24"

if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

import torch
import torch.nn as nn
import argparse
from torch.ao.quantization import QuantStub, DeQuantStub
from ultralytics import YOLO


def is_quantized_module(module):
    """Check if a module is quantized."""
    module_type = type(module).__name__
    module_path = type(module).__module__
    
    # Check for quantized module indicators
    is_quantized = (
        'quantized' in module_path.lower() or
        'Quantized' in module_type or
        hasattr(module, '_packed_params') and module._packed_params is not None
    )
    
    return is_quantized


def get_quantization_params(module):
    """Extract quantization parameters from a module."""
    params = {}
    
    if hasattr(module, 'scale'):
        scale = module.scale
        if isinstance(scale, torch.Tensor):
            params['scale'] = scale.item()
        else:
            params['scale'] = scale
    else:
        params['scale'] = None
    
    if hasattr(module, 'zero_point'):
        zp = module.zero_point
        if isinstance(zp, torch.Tensor):
            params['zero_point'] = zp.item()
        else:
            params['zero_point'] = zp
    else:
        params['zero_point'] = None
    
    if hasattr(module, '_packed_params'):
        params['has_packed_params'] = module._packed_params is not None
    else:
        params['has_packed_params'] = False
    
    return params


def inspect_botnet_modules(model):
    """Inspect BoTNet and MHSA modules specifically."""
    print("\n" + "=" * 80)
    print("BoTNet/MHSA Module Inspection")
    print("=" * 80)
    
    botnet_modules = []
    mhsa_modules = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if 'BoTNet' in module_type or 'botnet' in name.lower():
            botnet_modules.append((name, module))
        
        if 'MHSA' in module_type or 'mhsa' in name.lower():
            mhsa_modules.append((name, module))
    
    print(f"\nFound {len(botnet_modules)} BoTNet modules and {len(mhsa_modules)} MHSA modules")
    
    # Inspect BoTNet modules
    for name, module in botnet_modules:
        print(f"\n  BoTNet: {name}")
        print(f"    Type: {type(module).__name__}")
        print(f"    Module path: {type(module).__module__}")
        
        # Check submodules
        if hasattr(module, 'cv1'):
            cv1_type = type(module.cv1).__name__
            cv1_path = type(module.cv1).__module__
            cv1_quantized = is_quantized_module(module.cv1)
            print(f"    cv1: {cv1_type} ({cv1_path}) - Quantized: {cv1_quantized}")
            
            # If it's a Conv wrapper, check the inner conv
            if hasattr(module.cv1, 'conv'):
                inner_conv = module.cv1.conv
                inner_quantized = is_quantized_module(inner_conv)
                inner_type = type(inner_conv).__name__
                inner_path = type(inner_conv).__module__
                print(f"      cv1.conv: {inner_type} ({inner_path}) - Quantized: {inner_quantized}")
                if inner_quantized:
                    params = get_quantization_params(inner_conv)
                    print(f"        Scale: {params['scale']}, Zero-point: {params['zero_point']}")
        
        if hasattr(module, 'cv2'):
            cv2_type = type(module.cv2).__name__
            cv2_path = type(module.cv2).__module__
            cv2_quantized = is_quantized_module(module.cv2)
            print(f"    cv2: {cv2_type} ({cv2_path}) - Quantized: {cv2_quantized}")
            
            if hasattr(module.cv2, 'conv'):
                inner_conv = module.cv2.conv
                inner_quantized = is_quantized_module(inner_conv)
                inner_type = type(inner_conv).__name__
                inner_path = type(inner_conv).__module__
                print(f"      cv2.conv: {inner_type} ({inner_path}) - Quantized: {inner_quantized}")
                if inner_quantized:
                    params = get_quantization_params(inner_conv)
                    print(f"        Scale: {params['scale']}, Zero-point: {params['zero_point']}")
        
        if hasattr(module, 'cv3'):
            cv3_type = type(module.cv3).__name__
            cv3_path = type(module.cv3).__module__
            cv3_quantized = is_quantized_module(module.cv3)
            print(f"    cv3: {cv3_type} ({cv3_path}) - Quantized: {cv3_quantized}")
            
            if hasattr(module.cv3, 'conv'):
                inner_conv = module.cv3.conv
                inner_quantized = is_quantized_module(inner_conv)
                inner_type = type(inner_conv).__name__
                inner_path = type(inner_conv).__module__
                print(f"      cv3.conv: {inner_type} ({inner_path}) - Quantized: {inner_quantized}")
                if inner_quantized:
                    params = get_quantization_params(inner_conv)
                    print(f"        Scale: {params['scale']}, Zero-point: {params['zero_point']}")
    
    # Inspect MHSA modules
    for name, module in mhsa_modules:
        print(f"\n  MHSA: {name}")
        print(f"    Type: {type(module).__name__}")
        print(f"    Module path: {type(module).__module__}")
        
        # Check query, key, value Conv2d layers
        for conv_name in ['query', 'key', 'value']:
            if hasattr(module, conv_name):
                conv = getattr(module, conv_name)
                conv_type = type(conv).__name__
                conv_path = type(conv).__module__
                conv_quantized = is_quantized_module(conv)
                print(f"    {conv_name}: {conv_type} ({conv_path}) - Quantized: {conv_quantized}")
                
                if conv_quantized:
                    params = get_quantization_params(conv)
                    print(f"      Scale: {params['scale']}, Zero-point: {params['zero_point']}")
                    print(f"      Has _packed_params: {params['has_packed_params']}")
                else:
                    # Check if it's a regular Conv2d that should have been quantized
                    if isinstance(conv, nn.Conv2d):
                        print(f"      ⚠️  WARNING: Regular Conv2d found (should be quantized!)")
    
    return botnet_modules, mhsa_modules


def inspect_model_conversion(model):
    """Inspect overall model conversion state."""
    print("\n" + "=" * 80)
    print("Model Conversion State")
    print("=" * 80)
    
    # Check QuantStub/DeQuantStub
    has_quant = hasattr(model.model, 'quant') and isinstance(model.model.quant, QuantStub)
    has_dequant = hasattr(model.model, 'dequant') and isinstance(model.model.dequant, DeQuantStub)
    
    print(f"\nQuantStub: {'✓ Present' if has_quant else '✗ Missing'}")
    print(f"DeQuantStub: {'✓ Present' if has_dequant else '✗ Missing'}")
    
    # Check backend engine
    backend = torch.backends.quantized.engine
    print(f"\nQuantization Backend Engine: {backend}")
    print(f"Supported engines: {torch.backends.quantized.supported_engines}")
    
    # Count quantized vs FP32 operations
    quantized_count = 0
    fp32_conv_count = 0
    fp32_linear_count = 0
    observer_count = 0
    
    from torch.ao.quantization import ObserverBase
    
    for name, module in model.model.named_modules():
        module_type = type(module).__name__
        module_path = type(module).__module__
        
        if isinstance(module, ObserverBase):
            observer_count += 1
        
        if is_quantized_module(module):
            quantized_count += 1
        
        if isinstance(module, nn.Conv2d) and 'quantized' not in module_path.lower():
            fp32_conv_count += 1
        
        if isinstance(module, nn.Linear) and 'quantized' not in module_path.lower():
            fp32_linear_count += 1
    
    print(f"\nModule Statistics:")
    print(f"  Quantized operations: {quantized_count}")
    print(f"  FP32 Conv2d: {fp32_conv_count}")
    print(f"  FP32 Linear: {fp32_linear_count}")
    print(f"  Observers (should be 0): {observer_count}")
    
    if observer_count > 0:
        print(f"  ⚠️  WARNING: {observer_count} observers still present (conversion may be incomplete)")
    
    return {
        'has_quant': has_quant,
        'has_dequant': has_dequant,
        'backend': backend,
        'quantized_count': quantized_count,
        'fp32_conv_count': fp32_conv_count,
        'observer_count': observer_count
    }


def get_tensor_backend(tensor):
    """Get the backend of a tensor (CPU, QuantizedCPU, etc.)."""
    if hasattr(tensor, 'q_scale'):
        # Quantized tensor
        return f"QuantizedCPU (dtype={tensor.dtype}, scale={tensor.q_scale}, zp={tensor.q_zero_point})"
    else:
        # Regular tensor
        return f"{tensor.device.type.upper()} (dtype={tensor.dtype})"


def test_forward_pass(model, imgsz=640):
    """Test a single forward pass to identify failure point."""
    print("\n" + "=" * 80)
    print("Forward Pass Test")
    print("=" * 80)
    
    model.model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Input dtype: {dummy_input.dtype}")
    print(f"Input device: {dummy_input.device}")
    print(f"Input backend: {get_tensor_backend(dummy_input)}")
    
    # Check if model has QuantStub
    if hasattr(model.model, 'quant') and isinstance(model.model.quant, QuantStub):
        print("\nQuantizing input via QuantStub...")
        try:
            x_quant = model.model.quant(dummy_input)
            print(f"  Quantized input dtype: {x_quant.dtype}")
            print(f"  Quantized input device: {x_quant.device}")
            print(f"  Quantized input backend: {get_tensor_backend(x_quant)}")
            if hasattr(x_quant, 'q_scale'):
                print(f"  Quantized input scale: {x_quant.q_scale}")
                print(f"  Quantized input zero_point: {x_quant.q_zero_point}")
            
            # Try forward pass
            print("\nAttempting forward pass...")
            try:
                with torch.no_grad():
                    output = model.model(x_quant)
                print("  ✓ Forward pass succeeded!")
                return True
            except Exception as e:
                print(f"  ✗ Forward pass failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            print(f"  ✗ Quantization failed: {type(e).__name__}: {e}")
            return False
    else:
        print("\n⚠️  No QuantStub found - attempting direct forward pass...")
        try:
            with torch.no_grad():
                output = model.model(dummy_input)
            print("  ✓ Forward pass succeeded!")
            return True
        except Exception as e:
            print(f"  ✗ Forward pass failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Debug quantized backend dispatch issues')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to INT8 checkpoint')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for testing')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Quantized Backend Debug Tool")
    print("=" * 80)
    
    # Load checkpoint
    checkpoint_path = Path(args.weights)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get backend and set it
    backend = checkpoint.get('backend', 'fbgemm')
    print(f"Checkpoint backend: {backend}")
    
    if backend in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = backend
        print(f"✓ Set quantization backend engine to {backend}")
    else:
        print(f"⚠️  Backend '{backend}' not supported")
    
    # Load model structure
    yaml_config = checkpoint.get('yaml', {})
    if isinstance(yaml_config, dict):
        yaml_path = yaml_config.get('yaml_file', 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml')
    else:
        yaml_path = yaml_config
    
    if not Path(yaml_path).is_absolute():
        repo_root = Path(__file__).parent
        yaml_path = repo_root / yaml_path
        if not yaml_path.exists():
            yaml_path = repo_root / 'obc-yolov8' / 'ultralytics10.24' / 'ultralytics' / 'cfg' / 'models' / 'v8' / 'yolov8-CA.yaml'
    
    print(f"Loading model structure from: {yaml_path}")
    model = YOLO(str(yaml_path))
    model.model = checkpoint['model']
    
    # Repair bookkeeping
    print("\nRepairing model bookkeeping...")
    try:
        from ultralytics.nn.tasks import ensure_module_bookkeeping
        ensure_module_bookkeeping(model.model, recursive=True)
        
        hook_attrs = ('_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_backward_pre_hooks',
                     '_state_dict_hooks', '_load_state_dict_pre_hooks')
        
        for name, module in model.model.named_modules():
            if isinstance(module, nn.Module):
                for hook_attr in hook_attrs:
                    try:
                        attr_value = getattr(module, hook_attr)
                        if not isinstance(attr_value, dict):
                            object.__setattr__(module, hook_attr, {})
                    except AttributeError:
                        object.__setattr__(module, hook_attr, {})
                
                if not hasattr(module, '_non_persistent_buffers_set'):
                    object.__setattr__(module, '_non_persistent_buffers_set', set())
        
        print("✓ Bookkeeping repaired")
    except Exception as e:
        print(f"⚠️  Bookkeeping repair failed: {e}")
    
    # Run inspections
    conversion_state = inspect_model_conversion(model)
    botnet_modules, mhsa_modules = inspect_botnet_modules(model)
    forward_success = test_forward_pass(model, args.imgsz)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"QuantStub present: {conversion_state['has_quant']}")
    print(f"Backend engine: {conversion_state['backend']}")
    print(f"Quantized operations: {conversion_state['quantized_count']}")
    print(f"FP32 Conv2d remaining: {conversion_state['fp32_conv_count']}")
    print(f"Observers remaining: {conversion_state['observer_count']}")
    print(f"Forward pass: {'✓ Success' if forward_success else '✗ Failed'}")
    
    if not forward_success:
        print("\n⚠️  Forward pass failed - check BoTNet/MHSA modules above for unquantized Conv2d layers")


if __name__ == '__main__':
    main()

