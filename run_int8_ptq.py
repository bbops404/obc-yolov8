"""
Load and run inference with PTQ INT8 model.

Usage:
    python run_int8_ptq.py --weights runs/detect/train_ptq/weights/best_int8.pt --source path/to/images
    python run_int8_ptq.py --weights runs/detect/train_ptq/weights/best_int8.pt --source path/to/video.mp4
"""

import sys
from pathlib import Path
import time

# Add ultralytics path to sys.path (same as train_ptq.py)
REPO_ROOT = Path(__file__).parent.resolve()
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if not ULTRALYTICS_PATH.exists():
    ULTRALYTICS_PATH = REPO_ROOT / "ultralytics10.24"

if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

import torch
from torch.backends import quantized as torch_quantized_backends
import argparse

# Import YOLO - we'll patch the loader function when we need it
from ultralytics import YOLO

def load_ptq_int8_model(checkpoint_path, imgsz=640):
    """
    Load PTQ INT8 model from checkpoint.
    
    Uses the same simple loading approach as ablation_ptq.py which works correctly.
    
    Args:
        checkpoint_path: Path to INT8 checkpoint (.pt file)
        imgsz: Image size
    
    Returns:
        Loaded YOLO model ready for inference
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading INT8 checkpoint from {checkpoint_path}...")
    
    # Ensure some quantization engine is selected before torch.load tries to restore packed params
    default_backend = 'qnnpack'
    if default_backend in torch_quantized_backends.supported_engines:
        torch_quantized_backends.engine = default_backend
    
    # Load checkpoint metadata to get backend
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get backend from checkpoint and set it
    backend = checkpoint.get('backend', 'qnnpack')
    print(f"Backend: {backend}")
    
    # CRITICAL: Set quantization backend engine BEFORE any operations
    if backend in torch_quantized_backends.supported_engines:
        torch_quantized_backends.engine = backend
        print(f"✓ Set quantization backend engine to {backend}")
    else:
        print(f"⚠️  Backend '{backend}' not supported, using default")
    
    # YOLO's built-in loader doesn't work with quantized models (tries to call .float())
    # Use manual loading instead (like the old code, but simpler)
    print("Loading model manually (YOLO loader doesn't support quantized models)...")
    
    # Get YAML file path from checkpoint
    yaml_config = checkpoint.get('yaml', {})
    if isinstance(yaml_config, dict):
        yaml_path = yaml_config.get('yaml_file', 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml')
    else:
        yaml_path = yaml_config
    
    # Construct full path if relative
    if not Path(yaml_path).is_absolute():
        repo_root = Path(__file__).parent
        yaml_path = repo_root / yaml_path
        if not yaml_path.exists():
            yaml_path = repo_root / 'obc-yolov8' / 'ultralytics10.24' / 'ultralytics' / 'cfg' / 'models' / 'v8' / 'yolov8-CA.yaml'
    
    print(f"Loading model structure from: {yaml_path}")
    model = YOLO(str(yaml_path))
    
    # Replace the model with the quantized one from checkpoint
    model.model = checkpoint['model']
    
    # Repair bookkeeping for quantized modules
    print("Repairing quantized model bookkeeping...")
    try:
        from ultralytics.nn import tasks as ultralytics_tasks  # type: ignore[attr-defined]
        ensure_module_bookkeeping = getattr(
            ultralytics_tasks,
            "ensure_module_bookkeeping",
            lambda *args, **kwargs: None,
        )
        ensure_module_bookkeeping(model.model, recursive=True)

        # Fix hook attributes
        import torch.nn as nn
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
        
        print("✓ Repaired model bookkeeping")
    except Exception as e:
        print(f"⚠️  Could not repair bookkeeping: {e}")
    
    print("✓ INT8 model loaded successfully")
    
    # CRITICAL: Check if model has QuantStub and patch forward pass to use it
    from torch.ao.quantization import QuantStub, DeQuantStub
    
    has_quant = hasattr(model.model, 'quant') and isinstance(model.model.quant, QuantStub)
    has_dequant = hasattr(model.model, 'dequant') and isinstance(model.model.dequant, DeQuantStub)
    
    if has_quant:
        print(f"✓ Model has QuantStub - will quantize inputs before forward pass")
        
        # Patch _predict_once to use QuantStub
        original_predict_once = model.model._predict_once
        
        def patched_predict_once(self, x, profile=False, visualize=False):
            """Patched _predict_once that quantizes input for quantized models."""
            # Quantize input first
            x_quant = self.quant(x)
            # Call original _predict_once with quantized input
            # original_predict_once is already bound, so call it directly without self
            out = original_predict_once(x_quant, profile, visualize)
            # Dequantize output if DeQuantStub exists
            if hasattr(self, 'dequant') and isinstance(self.dequant, DeQuantStub):
                if isinstance(out, torch.Tensor):
                    out = self.dequant(out)
                elif isinstance(out, (list, tuple)):
                    out = tuple(self.dequant(o) if isinstance(o, torch.Tensor) else o for o in out)
            return out
        
        # Replace the method - bind it to model.model
        import types
        model.model._predict_once = types.MethodType(patched_predict_once, model.model)
        print("✓ Patched _predict_once to use QuantStub for quantized inputs")
    else:
        print("⚠️  Model does NOT have QuantStub - quantized operations may fail")
        print("   This is likely why you're getting the backend error")
        print("   The model may need to be re-converted with PTQ")
    
    # Verify BoTNet/MHSA modules are properly set up for quantized operations
    # The MHSA.forward() method has been patched to handle quantized Conv2d outputs
    print("\nVerifying BoTNet/MHSA module quantization...")
    try:
        from ultralytics.nn.BoTNet import MHSA, BoTNet, BottleneckTransformer
        
        botnet_count = 0
        mhsa_count = 0
        quantized_conv_in_mhsa = 0
        fp32_conv_in_mhsa = 0
        quantized_conv_in_botnet = 0
        fp32_conv_in_botnet = 0
        quantized_params_ok = 0
        quantized_params_default = 0
        
        def check_conv_quantization(conv_module, name_prefix=""):
            """Check if a Conv2d module is quantized and return status."""
            if conv_module is None:
                return None, None
            
            module_path = type(conv_module).__module__
            is_quantized = 'quantized' in module_path.lower() or hasattr(conv_module, '_packed_params')
            
            if is_quantized:
                # Check quantization parameters
                scale = None
                zp = None
                if hasattr(conv_module, 'scale') and hasattr(conv_module, 'zero_point'):
                    scale = conv_module.scale.item() if isinstance(conv_module.scale, torch.Tensor) else conv_module.scale
                    zp = conv_module.zero_point.item() if isinstance(conv_module.zero_point, torch.Tensor) else conv_module.zero_point
                return True, (scale, zp)
            else:
                return False, None
        
        for name, module in model.model.named_modules():
            if isinstance(module, BoTNet):
                botnet_count += 1
                # Check cv1, cv2, cv3 (these are Conv wrappers, check inner conv)
                for cv_name in ['cv1', 'cv2', 'cv3']:
                    if hasattr(module, cv_name):
                        cv_wrapper = getattr(module, cv_name)
                        if hasattr(cv_wrapper, 'conv'):
                            is_quant, params = check_conv_quantization(cv_wrapper.conv, f"{name}.{cv_name}")
                            if is_quant:
                                quantized_conv_in_botnet += 1
                                if params and params[0] is not None:
                                    if params[0] == 1.0 and params[1] == 0:
                                        quantized_params_default += 1
                                    else:
                                        quantized_params_ok += 1
                            else:
                                fp32_conv_in_botnet += 1
            elif isinstance(module, MHSA):
                mhsa_count += 1
                # Check if query/key/value are quantized
                for conv_name in ['query', 'key', 'value']:
                    if hasattr(module, conv_name):
                        conv = getattr(module, conv_name)
                        is_quant, params = check_conv_quantization(conv, f"{name}.{conv_name}")
                        
                        if is_quant:
                            quantized_conv_in_mhsa += 1
                            if params and params[0] is not None:
                                if params[0] == 1.0 and params[1] == 0:
                                    quantized_params_default += 1
                                else:
                                    quantized_params_ok += 1
                        else:
                            fp32_conv_in_mhsa += 1
        
        print(f"  Found {botnet_count} BoTNet modules, {mhsa_count} MHSA modules")
        print(f"  BoTNet Conv2d: {quantized_conv_in_botnet} quantized, {fp32_conv_in_botnet} FP32")
        print(f"  MHSA Conv2d: {quantized_conv_in_mhsa} quantized, {fp32_conv_in_mhsa} FP32")
        
        if fp32_conv_in_mhsa > 0 or fp32_conv_in_botnet > 0:
            print(f"  ⚠️  Found {fp32_conv_in_mhsa + fp32_conv_in_botnet} FP32 Conv2d (should be quantized)")
        
        total_quantized = quantized_conv_in_mhsa + quantized_conv_in_botnet
        if total_quantized > 0:
            print(f"  Quantization params: {quantized_params_ok} OK, {quantized_params_default} using defaults")
            if quantized_params_default > 0:
                print(f"  ⚠️  {quantized_params_default} quantized Conv2d using default scale=1.0, zp=0")
            print("  ✓ Quantized Conv2d found - forward() methods will handle dequantization")
        elif (mhsa_count > 0 or botnet_count > 0) and total_quantized == 0:
            print("  ⚠️  BoTNet/MHSA modules found but no quantized Conv2d detected")
    except Exception as e:
        print(f"  ⚠️  Could not verify BoTNet modules: {e}")
        import traceback
        traceback.print_exc()
    
    return model
    
    # OLD MANUAL LOADING CODE (kept for reference but not used)
    # Check if checkpoint has full model or state_dict
    if False and 'model' in checkpoint:
        # Full model object - load directly
        print("Checkpoint contains full model object, loading directly...")
        
        # Get YAML file path from checkpoint
        yaml_config = checkpoint.get('yaml', {})
        if isinstance(yaml_config, dict):
            # yaml is a dict, extract the yaml_file path
            yaml_path = yaml_config.get('yaml_file', 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml')
        else:
            # yaml is already a file path string
            yaml_path = yaml_config
        
        # Construct full path if relative
        if not Path(yaml_path).is_absolute():
            # Try relative to repo root
            repo_root = Path(__file__).parent
            yaml_path = repo_root / yaml_path
            if not yaml_path.exists():
                # Fallback to default
                yaml_path = repo_root / 'obc-yolov8' / 'ultralytics10.24' / 'ultralytics' / 'cfg' / 'models' / 'v8' / 'yolov8-CA.yaml'
        
        print(f"Loading model structure from: {yaml_path}")
        model = YOLO(str(yaml_path))
        model.model = checkpoint['model']
        
        # CRITICAL: Repair bookkeeping for quantized modules
        # Quantized Conv2d modules don't have _backward_pre_hooks, need to fix this
        print("Repairing quantized model bookkeeping...")
        try:
            from ultralytics.nn.tasks import ensure_module_bookkeeping
            
            # First ensure basic bookkeeping
            ensure_module_bookkeeping(model.model, recursive=True)
            
            # Then manually fix hook attributes that quantized modules might be missing
            # PyTorch's _call_impl checks for these attributes and raises AttributeError if missing
            import torch.nn as nn
            hook_attrs = ('_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_backward_pre_hooks',
                         '_state_dict_hooks', '_load_state_dict_pre_hooks')
            
            for name, module in model.model.named_modules():
                if isinstance(module, nn.Module):
                    # Initialize missing hook attributes as empty dicts
                    for hook_attr in hook_attrs:
                        try:
                            # Try to get attribute - if it doesn't exist, __getattr__ will raise AttributeError
                            attr_value = getattr(module, hook_attr)
                            # If it exists but isn't a dict, replace it
                            if not isinstance(attr_value, dict):
                                object.__setattr__(module, hook_attr, {})
                        except AttributeError:
                            # Attribute doesn't exist - create it using object.__setattr__ to bypass __setattr__
                            object.__setattr__(module, hook_attr, {})
                    
                    # Also ensure _non_persistent_buffers_set exists
                    if not hasattr(module, '_non_persistent_buffers_set'):
                        object.__setattr__(module, '_non_persistent_buffers_set', set())
            
            print("✓ Repaired model bookkeeping")
        except Exception as e:
            print(f"⚠️  Could not repair bookkeeping: {e}")
            print("   Model may still work, but some operations might fail")
        
        # Ensure model is in eval mode
        try:
            model.model.eval()
        except:
            pass  # Quantized models may not support .eval()
        
        print("✓ INT8 model loaded from full model object")
        return model
    
    elif 'model_state_dict' in checkpoint:
        # State dict - need to reconstruct model structure
        print("Checkpoint contains state_dict, reconstructing model structure...")
        
        # Load model structure from YAML
        yaml_path = checkpoint.get('yaml')
        if yaml_path is None:
            # Try default path
            yaml_path = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
        
        model = YOLO(yaml_path)
        
        # Get example input size
        ch = model.model.yaml.get('ch', 3)
        example_input = torch.randn(1, ch, imgsz, imgsz)
        
        # Prepare for PTQ (inserts observers)
        print("Preparing model for PTQ structure...")
        model.model = model.model.prepare_for_ptq(
            backend=backend,
            example_input=example_input,
            use_fx=False,  # Use eager mode for compatibility
        )
        
        # Convert to INT8 (replaces observers with quantized operations)
        print("Converting to INT8...")
        model.model = model.model.convert_ptq_to_int8(backend=backend)
        
        # Load the INT8 state_dict
        print("Loading INT8 weights from state_dict...")
        model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # CRITICAL: Repair bookkeeping for quantized modules
        print("Repairing quantized model bookkeeping...")
        try:
            from ultralytics.nn.tasks import ensure_module_bookkeeping
            
            # First ensure basic bookkeeping
            ensure_module_bookkeeping(model.model, recursive=True)
            
            # Then manually fix hook attributes that quantized modules might be missing
            # PyTorch's _call_impl checks for these attributes and raises AttributeError if missing
            import torch.nn as nn
            hook_attrs = ('_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_backward_pre_hooks',
                         '_state_dict_hooks', '_load_state_dict_pre_hooks')
            
            for name, module in model.model.named_modules():
                if isinstance(module, nn.Module):
                    # Initialize missing hook attributes as empty dicts
                    for hook_attr in hook_attrs:
                        try:
                            attr_value = getattr(module, hook_attr)
                            if not isinstance(attr_value, dict):
                                object.__setattr__(module, hook_attr, {})
                        except AttributeError:
                            object.__setattr__(module, hook_attr, {})
                    
                    # Also ensure _non_persistent_buffers_set exists
                    if not hasattr(module, '_non_persistent_buffers_set'):
                        object.__setattr__(module, '_non_persistent_buffers_set', set())
            
            print("✓ Repaired model bookkeeping")
        except Exception as e:
            print(f"⚠️  Could not repair bookkeeping: {e}")
        
        # Ensure model is in eval mode
        try:
            model.model.eval()
        except:
            pass  # Quantized models may not support .eval()
        
        print("✓ INT8 model loaded from state_dict")
        return model
    
    else:
        raise ValueError("Checkpoint must contain either 'model' or 'model_state_dict'")


def main():
    parser = argparse.ArgumentParser(description='Run inference with PTQ INT8 model')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to INT8 checkpoint (e.g., runs/detect/train_ptq/weights/best_int8.pt)')
    parser.add_argument('--source', type=str, required=False,
                       help='Source for inference (image, video, folder, or webcam). Required for --mode predict')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu, fbgemm/qnnpack require CPU)')
    parser.add_argument('--save', action='store_true',
                       help='Save inference results')
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'val'],
                       help='Mode: predict (inference) or val (evaluation with metrics)')
    parser.add_argument('--data', type=str, default=None,
                       help='Dataset YAML for validation mode (required if --mode val)')
    parser.add_argument('--quant-stats', action='store_true',
                       help='Enable quantization statistics (shows quantized vs FP32 fallback operations)')
    
    args = parser.parse_args()
    
    
    # Validate arguments based on mode
    if args.mode == 'predict' and not args.source:
        parser.error("--source is required when using --mode predict")
    if args.mode == 'val' and not args.data:
        parser.error("--data is required when using --mode val")
    
    # Enable quantization statistics if requested
    if args.quant_stats:
        import os
        os.environ['ENABLE_QUANT_STATS'] = '1'
        # Directly enable statistics tracking (module may already be imported)
        try:
            import ultralytics.nn.modules.conv as conv_module  # type: ignore[attr-defined]
            _ENABLE_QUANT_STATS = getattr(conv_module, "_ENABLE_QUANT_STATS", None)
            _QUANTIZATION_STATS = getattr(conv_module, "_QUANTIZATION_STATS", None)
            if isinstance(_ENABLE_QUANT_STATS, dict):
                _ENABLE_QUANT_STATS.clear()
                _ENABLE_QUANT_STATS["enabled"] = True
            if isinstance(_QUANTIZATION_STATS, dict):
                _QUANTIZATION_STATS.clear()
            else:
                setattr(conv_module, "_QUANTIZATION_STATS", {})
            print("✓ Quantization statistics tracking enabled")
        except (ImportError, AttributeError) as e:
            print(f"⚠️  Warning: Could not enable quantization statistics: {e}")
    
    # Load INT8 model
    model = load_ptq_int8_model(args.weights, imgsz=args.imgsz)
    
    # Set device (fbgemm/qnnpack backends require CPU)
    if args.device != 'cpu':
        print("⚠️  Warning: fbgemm/qnnpack backends require CPU, switching to CPU")
        args.device = 'cpu'
    
    # Run inference or evaluation
    if args.mode == 'val':
        # Evaluation mode (like ablation_ptq.py)
        print(f"\nRunning evaluation on dataset: {args.data}...")
        results = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=16,
            device=args.device,
            plots=False,
            save=args.save,
            verbose=True,
        )
        print(f"\n✓ Evaluation complete!")
        if hasattr(results, 'map50'):
            print(f"  mAP@0.5: {results.map50:.4f}")
        if hasattr(results, 'map'):
            print(f"  mAP@0.5:0.95: {results.map:.4f}")
    else:
        # Inference mode (default)
        print(f"\nRunning inference on {args.source}...")
        results = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            save=args.save,
        )
        # Handle both list and DetMetrics objects
        if isinstance(results, list):
            print(f"\n✓ Inference complete! Processed {len(results)} image(s)")
        else:
            print(f"\n✓ Inference complete!")
    
    # Set device (fbgemm/qnnpack backends require CPU)
    if args.device != 'cpu':
        print("⚠️  Warning: fbgemm/qnnpack backends require CPU, switching to CPU")
        args.device = 'cpu'


    # This attempts to trick YOLOv8's AutoBackend into not running fusion logic
    if hasattr(model, 'is_fused'):
        model.is_fused = True # Set a flag that prevents AutoBackend from fusing
# ---
    
    # Run inference or evaluation
    if args.mode == 'val':
        # Evaluation mode (like ablation_ptq.py)
        # ... (val code remains the same) ...
        print(f"\nRunning evaluation on dataset: {args.data}...")
        results = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=16,
            device=args.device,
            plots=False,
            save=args.save,
            verbose=True,
        )
        print(f"\n✓ Evaluation complete!")
        if hasattr(results, 'map50'):
            print(f"  mAP@0.5: {results.map50:.4f}")
        if hasattr(results, 'map'):
            print(f"  mAP@0.5:0.95: {results.map:.4f}")
    else:
        # Inference mode (default) - ADDING BENCHMARKING
        print(f"\nRunning inference on {args.source} for speed test...")
        
        # 1. Load the first image/frame to process and warm up the model
        print("Warming up model with one run...")
        # Note: YOLO.predict returns a generator/list of results
        warmup_results = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            save=False, # Don't save warmup output
            stream=True, # Use stream mode for efficient single-image processing
        )
        # Process the results (need to iterate through generator)
        try:
            _ = next(iter(warmup_results))
        except StopIteration:
            print("⚠️  No detections found in warmup run. Cannot proceed with speed test.")
            return

        # 2. Run benchmark loop
        start_time = time.perf_counter()
        
        # We need a predictable number of frames/images to process.
        # This requires the source to be a single image or video file
        total_runs = args.runs
        
        print(f"Benchmarking {total_runs} runs...")
        for i in range(total_runs):
             # Run prediction using stream mode for speed
            results = model.predict(
                source=args.source,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                save=False,
                stream=True,
            )
            # Must iterate through the results generator to force execution
            try:
                _ = next(iter(results))
            except StopIteration:
                # This should not happen if the source is valid
                print("Error: Generator stopped unexpectedly during benchmark.")
                total_runs = i + 1
                break
        
        end_time = time.perf_counter()
        
        # 3. Calculate metrics
        total_time_s = end_time - start_time
        avg_latency_ms = (total_time_s / total_runs) * 1000
        fps = total_runs / total_time_s
        
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*50)
        print(f"Source: {args.source}")
        print(f"Backend: {model.model.backend}")
        print(f"Total runs: {total_runs}")
        print(f"Total time: {total_time_s:.3f} seconds")
        print("---")
        print(f"🖼️  Average Latency: {avg_latency_ms:.2f} ms")
        print(f"⏱️  Inference Speed (FPS): {fps:.2f} FPS")
        print("="*50)

    # Print quantization statistics if enabled
    if args.quant_stats:
        try:
            import ultralytics.nn.modules.conv as conv_module  # type: ignore[attr-defined]
            _ENABLE_QUANT_STATS = getattr(conv_module, "_ENABLE_QUANT_STATS", {})
            _QUANTIZATION_STATS = getattr(conv_module, "_QUANTIZATION_STATS", {})
            # Always print stats if enabled, even if empty (to show it's working)
            print("\n" + "="*60)
            print("QUANTIZATION OPERATION STATISTICS")
            print("="*60)
            print(f"Statistics tracking enabled: {_ENABLE_QUANT_STATS}")
            print(f"Statistics dict entries: {len(_QUANTIZATION_STATS)}")
            
            total_quantized = _QUANTIZATION_STATS.get('quantized_conv2d_success', 0)
            total_fallback = _QUANTIZATION_STATS.get('quantized_conv2d_fallback', 0)
            total_regular = _QUANTIZATION_STATS.get('regular_conv2d', 0)
            
            if total_regular + total_quantized + total_fallback > 0:
                print(f"\nRegular Conv2d (FP32):           {total_regular:>6}")
                print(f"Quantized Conv2d (INT8):          {total_quantized:>6} ✓")
                print(f"Quantized Conv2d (FP32 fallback): {total_fallback:>6} ⚠️")
                print(f"Total Conv2d operations:          {total_regular + total_quantized + total_fallback:>6}")
                
                if total_quantized + total_fallback > 0:
                    quantized_pct = (total_quantized / (total_quantized + total_fallback)) * 100
                    fallback_pct = (total_fallback / (total_quantized + total_fallback)) * 100
                    print(f"\nQuantization Success Rate:    {quantized_pct:.1f}%")
                    print(f"Fallback Rate:                 {fallback_pct:.1f}%")
                
                # Show module-specific statistics
                module_stats = {k: v for k, v in _QUANTIZATION_STATS.items() 
                              if k.startswith('quantized_') and ('_success_' in k or '_fallback_' in k)}
                if module_stats:
                    print("\nTop modules by operation count:")
                    for key, count in sorted(module_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                        module_name = key.replace('quantized_success_', '').replace('quantized_fallback_', '')
                        status = "✓" if 'success' in key else "⚠️"
                        print(f"  {status} {module_name}: {count}")
            else:
                print("\n⚠️  No statistics collected. This may indicate:")
                print("  - Statistics tracking was not enabled during forward pass")
                print("  - No Conv2d operations were called")
                print("  - Module was imported before ENABLE_QUANT_STATS was set")
                if not _ENABLE_QUANT_STATS:
                    print("  - Statistics tracking flag is False")
            print("="*60)
        except ImportError as e:
            print(f"\n⚠️  Could not import quantization statistics: {e}")
    
    # Print results summary (only for inference mode, not validation)
    if args.mode == 'predict' and isinstance(results, list):
        for i, result in enumerate(results):
            print(f"\nImage {i+1}:")
            print(f"  Detections: {len(result.boxes)}")
            if len(result.boxes) > 0:
                print(f"  Classes: {result.boxes.cls.unique().tolist()}")
                print(f"  Confidences: {result.boxes.conf.min().item():.3f} - {result.boxes.conf.max().item():.3f}")


if __name__ == '__main__':
    main()

