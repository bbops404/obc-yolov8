#!/usr/bin/env python3
"""
Check if INT8 model has collected statistics from QAT training.
This script loads the INT8 checkpoint and verifies that FakeQuantize modules
have proper min/max statistics before conversion.
"""

import sys
from pathlib import Path
import torch
from torch.ao.quantization import FakeQuantize

# Add ultralytics to path
REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

def check_int8_stats(checkpoint_path: str):
    """Check if the INT8 checkpoint has proper statistics."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract model
        if 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Check checkpoint metadata
        if isinstance(checkpoint, dict):
            if 'qat' in checkpoint:
                print(f"  Checkpoint type: QAT (qat={checkpoint['qat']})")
            if 'int8' in checkpoint:
                print(f"  Checkpoint type: INT8 (int8={checkpoint['int8']})")
            if 'backend' in checkpoint:
                print(f"  Backend: {checkpoint['backend']}")
        
        # Check if this is a QAT model (before conversion) or INT8 model (after conversion)
        fakequant_modules = []
        quantized_modules = []
        
        # Use a safer recursive function to find FakeQuantize modules
        def find_fakequantize_safe(module, path="", max_depth=50, current_depth=0):
            """Safely find FakeQuantize modules without using named_modules()"""
            results = []
            if current_depth > max_depth:
                return results
            
            if isinstance(module, FakeQuantize):
                results.append((path, module))
            
            # Only recurse if module has _modules attribute (most nn.Modules do)
            if hasattr(module, '_modules') and module._modules is not None:
                for name, child in module._modules.items():
                    if child is not None:
                        child_path = f"{path}.{name}" if path else name
                        results.extend(find_fakequantize_safe(child, child_path, max_depth, current_depth + 1))
            
            return results
        
        try:
            # Try standard named_modules first
            for name, module in model.named_modules():
                if isinstance(module, FakeQuantize):
                    fakequant_modules.append((name, module))
                elif hasattr(module, '_packed_params') and module._packed_params is not None:
                    quantized_modules.append((name, module))
        except (AttributeError, RuntimeError) as e:
            # Some quantized modules (like QuantizedConv2d) don't have _modules
            # Try a different approach - check the model structure more carefully
            print(f"  ⚠️  Error iterating modules: {e}")
            print("   Trying alternative method to detect model type...")
            
            # Use safe recursive search
            try:
                fakequant_modules = find_fakequantize_safe(model)
                print(f"   Found {len(fakequant_modules)} FakeQuantize modules using safe search")
            except Exception as e2:
                print(f"   ❌ Could not analyze model structure: {e2}")
                # Check if model has quantized parameters as fallback
                has_quantized_params = False
                try:
                    for name, param in model.named_parameters():
                        if 'scale' in name.lower() or 'zero_point' in name.lower():
                            has_quantized_params = True
                            break
                except:
                    pass
                
                if has_quantized_params:
                    print("   ✓ Detected quantized parameters - this is an INT8 model")
                    print("   INT8 models don't have FakeQuantize modules - conversion already completed.")
                    return True
                else:
                    return False
        
        if fakequant_modules:
            print(f"\n⚠️  Found {len(fakequant_modules)} FakeQuantize modules (this is a QAT model, not INT8)")
            print("   Checking statistics in FakeQuantize modules...")
            
            uncalibrated = []
            calibrated = []
            has_nan = []
            
            for name, fq_module in fakequant_modules:
                try:
                    # Try to access observer statistics
                    observer = fq_module.activation_post_process if hasattr(fq_module, 'activation_post_process') else None
                    
                    if observer is None:
                        # Check if FakeQuantize has direct min/max
                        if hasattr(fq_module, 'scale') and hasattr(fq_module, 'zero_point'):
                            scale = fq_module.scale
                            zero_point = fq_module.zero_point
                            
                            if scale is not None and zero_point is not None:
                                if torch.isnan(scale).any() or torch.isnan(zero_point).any():
                                    has_nan.append(name)
                                else:
                                    calibrated.append(name)
                            else:
                                uncalibrated.append(name)
                        else:
                            uncalibrated.append(name)
                    else:
                        # Check observer statistics
                        min_val = getattr(observer, 'min_val', None)
                        max_val = getattr(observer, 'max_val', None)
                        
                        if min_val is None or max_val is None:
                            uncalibrated.append(name)
                        else:
                            # Check for NaN
                            if isinstance(min_val, torch.Tensor):
                                if torch.isnan(min_val).any() or torch.isnan(max_val).any():
                                    has_nan.append(name)
                                else:
                                    calibrated.append(name)
                            else:
                                if torch.isnan(torch.tensor(min_val)) or torch.isnan(torch.tensor(max_val)):
                                    has_nan.append(name)
                                else:
                                    calibrated.append(name)
                except Exception as e:
                    print(f"  ⚠️  Error checking {name}: {e}")
                    uncalibrated.append(name)
            
            print(f"\n  Statistics Summary:")
            print(f"    ✓ Calibrated: {len(calibrated)}")
            print(f"    ❌ Uncalibrated: {len(uncalibrated)}")
            print(f"    ⚠️  Has NaN: {len(has_nan)}")
            
            if has_nan:
                print(f"\n  ⚠️  Modules with NaN statistics:")
                for name in has_nan[:10]:  # Show first 10
                    print(f"      - {name}")
                if len(has_nan) > 10:
                    print(f"      ... and {len(has_nan) - 10} more")
            
            if uncalibrated:
                print(f"\n  ❌ Uncalibrated modules:")
                for name in uncalibrated[:10]:  # Show first 10
                    print(f"      - {name}")
                if len(uncalibrated) > 10:
                    print(f"      ... and {len(uncalibrated) - 10} more")
            
            return len(has_nan) == 0 and len(uncalibrated) == 0
        
        elif quantized_modules:
            print(f"\n✓ Found {len(quantized_modules)} quantized modules (this is an INT8 model)")
            print("   INT8 models don't have FakeQuantize - they're already converted.")
            print("   To check statistics, you need to check the QAT model before conversion.")
            return True
        
        else:
            print(f"\n⚠️  No FakeQuantize or quantized modules found")
            print("   This might be a FP32 model or an unexpected format.")
            return False
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check INT8 model statistics")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to INT8 checkpoint (or QAT checkpoint to check before conversion)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("INT8 Statistics Checker")
    print("=" * 80)
    
    result = check_int8_stats(args.checkpoint)
    
    print("\n" + "=" * 80)
    if result:
        print("✓ Statistics check PASSED")
    else:
        print("❌ Statistics check FAILED - some modules may be uncalibrated or have NaN")
    print("=" * 80)

