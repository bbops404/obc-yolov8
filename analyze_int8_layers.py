#!/usr/bin/env python3
"""
Utility script to analyze INT8 model and identify which layers are from PTQ vs QAT.

Usage:
    python analyze_int8_layers.py <path_to_int8_model.pt> [--sensitive-modules model.10,model.19]
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

def analyze_int8_model(checkpoint_path, sensitive_modules=None):
    """
    Analyze an INT8 model checkpoint to identify PTQ vs QAT layers.
    
    Args:
        checkpoint_path: Path to INT8 model checkpoint (.pt file)
        sensitive_modules: List of module patterns that were QAT (default: ['model.10'])
    
    Returns:
        Dictionary with analysis results
    """
    if sensitive_modules is None:
        sensitive_modules = ['model.10']  # Default: BoTNet
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("=" * 80)
    print(f"Analyzing INT8 Model: {checkpoint_path}")
    print("=" * 80)
    
    # Load checkpoint - try multiple methods
    checkpoint = None
    try:
        # First try: Load with weights_only=False (requires dill, but gets full model)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except (ModuleNotFoundError, ImportError, Exception) as e:
        # If that fails, try to extract just metadata using pickle with custom unpickler
        print(f"⚠️  Warning: Could not load full model ({type(e).__name__}: {e})")
        print("   Attempting to extract metadata only...")
        try:
            import pickle
            import sys
            
            # Create a custom unpickler that skips problematic objects
            class MetadataUnpickler(pickle.Unpickler):
                def persistent_load(self, pid):
                    # Skip persistent IDs (like tensor storage)
                    return None
                
                def load_global(self):
                    # Try to load global, but skip if it fails
                    try:
                        return super().load_global()
                    except:
                        # Return a dummy object for unknown classes
                        class Dummy:
                            pass
                        return Dummy
            
            with open(checkpoint_path, 'rb') as f:
                unpickler = MetadataUnpickler(f)
                try:
                    # Try to load - this might partially succeed
                    checkpoint = unpickler.load()
                    # Check if we got a dict with metadata
                    if not isinstance(checkpoint, dict):
                        checkpoint = None
                except Exception as load_err:
                    # If full load fails, try to manually extract dict keys
                    print(f"   Full unpickle failed: {load_err}")
                    print("   Trying alternative method...")
                    checkpoint = None
            
            # Alternative: Try using torch's _load with error handling
            if checkpoint is None:
                try:
                    # Use torch's internal loading but catch errors
                    import torch.serialization
                    with open(checkpoint_path, 'rb') as f:
                        # Read the pickle file and try to extract just the dict structure
                        # This is a workaround - we'll try to get metadata keys
                        f.seek(0)
                        # Try to load with a custom find_class that returns None for unknown classes
                        original_find_class = pickle.Unpickler.find_class
                        def safe_find_class(self, module, name):
                            try:
                                return original_find_class(self, module, name)
                            except (ImportError, AttributeError):
                                # Return a dummy class for unknown modules
                                class DummyClass:
                                    pass
                                return DummyClass
                        
                        pickle.Unpickler.find_class = safe_find_class
                        try:
                            unpickler = pickle.Unpickler(f)
                            checkpoint = unpickler.load()
                            if not isinstance(checkpoint, dict):
                                checkpoint = None
                        finally:
                            pickle.Unpickler.find_class = original_find_class
                except Exception as e2:
                    print(f"   Alternative method also failed: {e2}")
                    checkpoint = None
                    
        except Exception as e2:
            print(f"   Could not extract metadata: {e2}")
            checkpoint = None
    
    if checkpoint is None:
        print("\n❌ Could not load checkpoint.")
        print("   The checkpoint may need to be re-saved with metadata.")
        print("   For now, you can identify QAT vs PTQ layers by module name:")
        print("   - QAT layers: modules matching sensitive_modules patterns (default: 'model.10')")
        print("   - PTQ layers: all other quantized modules")
        return None
    
    # Check for metadata
    print("\nCheckpoint Metadata:")
    print("-" * 80)
    metadata_keys = ['int8', 'hybrid_qat', 'backend', 'epoch', 'date', 'sensitive_modules', 'qat_layers', 'ptq_layers']
    for key in metadata_keys:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, list) and len(value) > 10:
                print(f"  {key}: {len(value)} items (list)")
            else:
                print(f"  {key}: {value}")
    
    # Check if metadata already contains QAT/PTQ info
    if 'qat_layers' in checkpoint and 'ptq_layers' in checkpoint:
        print("\n" + "=" * 80)
        print("Layer Analysis: PTQ vs QAT (from checkpoint metadata)")
        print("=" * 80)
        
        qat_layers = checkpoint['qat_layers']
        ptq_layers = checkpoint['ptq_layers']
        stored_sensitive = checkpoint.get('sensitive_modules', sensitive_modules)
        
        print(f"\n📊 Summary:")
        print(f"  QAT layers (sensitive, fine-tuned):     {len(qat_layers)}")
        print(f"  PTQ layers (calibrated only):          {len(ptq_layers)}")
        
        print(f"\n🔧 QAT Layers (from sensitive modules: {', '.join(stored_sensitive)}):")
        print("-" * 80)
        if qat_layers:
            for layer in sorted(qat_layers)[:20]:
                print(f"  ✓ {layer}")
            if len(qat_layers) > 20:
                print(f"  ... and {len(qat_layers) - 20} more")
        else:
            print("  (none found)")
        
        print(f"\n📐 PTQ Layers (calibrated, not fine-tuned):")
        print("-" * 80)
        if ptq_layers:
            for layer in sorted(ptq_layers)[:20]:
                print(f"  • {layer}")
            if len(ptq_layers) > 20:
                print(f"  ... and {len(ptq_layers) - 20} more")
        else:
            print("  (none found)")
        
        return {
            'qat_layers': qat_layers,
            'ptq_layers': ptq_layers,
            'sensitive_modules': stored_sensitive
        }
    
    # Get model for analysis
    if 'model' in checkpoint:
        model = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        print("\n⚠️  Warning: Only state_dict found, cannot analyze module structure")
        print("   Need full model object to identify PTQ vs QAT layers")
        print("   However, if the checkpoint was saved with metadata, it should have qat_layers/ptq_layers")
        return
    else:
        print("\n❌ Error: No model found in checkpoint")
        return
    
    # Analyze modules
    print("\n" + "=" * 80)
    print("Layer Analysis: PTQ vs QAT")
    print("=" * 80)
    
    from torch.ao.quantization import FakeQuantize
    from torch.ao.quantization.observer import ObserverBase
    
    qat_layers = []
    ptq_layers = []
    fp32_layers = []
    quantized_layers = []
    
    # Check each module
    for name, module in model.named_modules():
        # Skip leaf modules that are not Conv2d-like
        if not (hasattr(module, 'in_channels') or isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))):
            continue
        
        # Check if it's a real quantized module (INT8)
        is_quantized = (
            hasattr(module, '_packed_params') or
            ('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
        )
        
        # Check if it has FakeQuantize (QAT) or Observer (PTQ) remnants
        has_fakequant = isinstance(module, FakeQuantize) or (
            hasattr(module, 'activation_post_process') and 
            isinstance(module.activation_post_process, FakeQuantize)
        )
        has_observer = (
            hasattr(module, 'activation_post_process') and 
            isinstance(module.activation_post_process, ObserverBase) and
            not isinstance(module.activation_post_process, FakeQuantize)
        )
        
        # Determine if this module was QAT or PTQ based on name pattern
        is_sensitive = any(pattern in name for pattern in sensitive_modules)
        
        if is_quantized:
            quantized_layers.append(name)
            if is_sensitive:
                qat_layers.append(name)
            else:
                ptq_layers.append(name)
        elif has_fakequant:
            qat_layers.append(name)
        elif has_observer:
            ptq_layers.append(name)
        else:
            # FP32 layer (not quantized)
            fp32_layers.append(name)
    
    # Print results
    print(f"\n📊 Summary:")
    print(f"  QAT layers (sensitive, fine-tuned):     {len(qat_layers)}")
    print(f"  PTQ layers (calibrated only):          {len(ptq_layers)}")
    print(f"  FP32 layers (not quantized):            {len(fp32_layers)}")
    print(f"  Total quantized layers:                {len(quantized_layers)}")
    
    print(f"\n🔧 QAT Layers (from sensitive modules: {', '.join(sensitive_modules)}):")
    print("-" * 80)
    if qat_layers:
        for layer in sorted(qat_layers)[:20]:
            print(f"  ✓ {layer}")
        if len(qat_layers) > 20:
            print(f"  ... and {len(qat_layers) - 20} more")
    else:
        print("  (none found)")
    
    print(f"\n📐 PTQ Layers (calibrated, not fine-tuned):")
    print("-" * 80)
    if ptq_layers:
        for layer in sorted(ptq_layers)[:20]:
            print(f"  • {layer}")
        if len(ptq_layers) > 20:
            print(f"  ... and {len(ptq_layers) - 20} more")
    else:
        print("  (none found)")
    
    print(f"\n🔷 FP32 Layers (not quantized):")
    print("-" * 80)
    if fp32_layers:
        for layer in sorted(fp32_layers)[:10]:
            print(f"  - {layer}")
        if len(fp32_layers) > 10:
            print(f"  ... and {len(fp32_layers) - 10} more")
    else:
        print("  (none found)")
    
    # Group by module type
    print(f"\n📦 Grouped by Module Pattern:")
    print("-" * 80)
    qat_by_pattern = defaultdict(list)
    ptq_by_pattern = defaultdict(list)
    
    for layer in qat_layers:
        # Extract pattern (e.g., 'model.10' from 'model.10.cv1.conv')
        parts = layer.split('.')
        if len(parts) >= 2:
            pattern = '.'.join(parts[:2])  # e.g., 'model.10'
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
        'fp32_layers': fp32_layers,
        'quantized_layers': quantized_layers,
        'sensitive_modules': sensitive_modules
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_int8_layers.py <path_to_int8_model.pt> [--sensitive-modules model.10,model.19]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    sensitive_modules = None
    
    # Parse sensitive modules if provided
    if '--sensitive-modules' in sys.argv:
        idx = sys.argv.index('--sensitive-modules')
        if idx + 1 < len(sys.argv):
            sensitive_modules = sys.argv[idx + 1].split(',')
    
    try:
        analyze_int8_model(checkpoint_path, sensitive_modules)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

