"""
Evaluate INT8 quantized YOLOv8-CA model
Compares performance metrics between FP32 and INT8 models
"""

import sys
from pathlib import Path

# Add local ultralytics to path BEFORE importing
sys.path.insert(0, str(Path(__file__).parent / 'obc-yolov8' / 'ultralytics10.24'))

import torch
import warnings
from ultralytics import YOLO
from ultralytics.nn.tasks import ensure_module_bookkeeping
from ultralytics.utils import LOGGER

warnings.filterwarnings('ignore')

# CRITICAL: Set default quantization backend early
# This must be set before any quantized operations are created
# Default to fbgemm for x86/CPU quantization
# Note: This will be overridden by the checkpoint's backend if different
try:
    # Try qnnpack first if available (may work better on some systems)
    if 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
    elif 'fbgemm' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'fbgemm'
except:
    pass  # Engine might not be available in all PyTorch builds


def _is_quantized_model(model):
    """
    Check if a model is quantized by looking for QuantizedConv2d or _packed_params.
    
    Args:
        model: PyTorch model to check
        
    Returns:
        bool: True if model appears to be quantized
    """
    try:
        # Try to iterate modules - quantized models may fail here
        for name, module in model.named_modules():
            module_type = type(module).__name__
            # Check for quantized module types
            if 'Quantized' in module_type:
                return True
            # Check for _packed_params (sign of quantization)
            if hasattr(module, '_packed_params'):
                return True
        return False
    except (AttributeError, RuntimeError):
        # If named_modules fails, it's likely a quantized model
        # QuantizedConv2d doesn't have _modules attribute
        return True


def _repair_quantized_bookkeeping(root_module):
    """Repair missing nn.Module bookkeeping attributes on quantized modules."""
    if root_module is None:
        return 0

    visited = set()
    stack = [root_module]
    modules_needing_fix = 0

    while stack:
        module = stack.pop()
        if not isinstance(module, torch.nn.Module):  # type: ignore[attr-defined]
            continue
        module_id = id(module)
        if module_id in visited:
            continue
        visited.add(module_id)

        def _is_dict_like(value):
            return isinstance(value, dict)

        needs_fix = False
        try:
            if not _is_dict_like(getattr(module, '_modules', None)):
                needs_fix = True
        except Exception:
            needs_fix = True
        try:
            if not _is_dict_like(getattr(module, '_parameters', None)):
                needs_fix = True
        except Exception:
            needs_fix = True
        try:
            if not _is_dict_like(getattr(module, '_buffers', None)):
                needs_fix = True
        except Exception:
            needs_fix = True

        # CRITICAL: Fix hook attributes - these must exist as dicts, not just be checked
        # PyTorch's _call_impl checks for these and raises AttributeError if missing
        hook_attrs = ('_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_backward_pre_hooks',
                     '_state_dict_hooks', '_load_state_dict_pre_hooks')
        
        for hook_attr in hook_attrs:
            try:
                attr_value = getattr(module, hook_attr)
                if not _is_dict_like(attr_value):
                    # Exists but not a dict - fix it
                    object.__setattr__(module, hook_attr, {})
                    needs_fix = True
            except AttributeError:
                # Attribute doesn't exist - create it using object.__setattr__ to bypass __setattr__
                object.__setattr__(module, hook_attr, {})
                needs_fix = True
            except Exception:
                needs_fix = True

        # Fix _non_persistent_buffers_set
        try:
            if not hasattr(module, '_non_persistent_buffers_set') or not isinstance(getattr(module, '_non_persistent_buffers_set'), set):
                object.__setattr__(module, '_non_persistent_buffers_set', set())
                needs_fix = True
        except Exception:
            object.__setattr__(module, '_non_persistent_buffers_set', set())
            needs_fix = True
        
        # Fix training attribute
        if not hasattr(module, 'training'):
            object.__setattr__(module, 'training', False)
            needs_fix = True

        if needs_fix:
            modules_needing_fix += 1

        try:
            children = getattr(module, '_modules', None)
            if isinstance(children, dict):
                stack.extend(child for child in children.values() if child is not None)
            else:
                stack.extend(list(module.children()))
        except Exception:
            pass

    ensure_module_bookkeeping(root_module, recursive=True)
    return modules_needing_fix


def convert_qat_checkpoint(
    qat_checkpoint_path: str,
    output_path: str,
    imgsz: int = 640,
):
    """Convert a QAT checkpoint into an INT8 checkpoint and persist it."""

    qat_path = Path(qat_checkpoint_path)
    if not qat_path.exists():
        raise FileNotFoundError(f"QAT checkpoint not found: {qat_checkpoint_path}")

    LOGGER.info("=" * 80)
    LOGGER.info(f"Converting QAT checkpoint to INT8: {qat_checkpoint_path}")
    LOGGER.info("=" * 80)

    checkpoint = torch.load(qat_path, map_location='cpu', weights_only=False)
    backend = checkpoint.get('backend', 'fbgemm')
    if backend in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = backend
    else:
        LOGGER.warning(f"Backend '{backend}' not supported, defaulting to existing engine '{torch.backends.quantized.engine}'")

    if 'model' not in checkpoint:
        raise KeyError(f"Checkpoint at {qat_checkpoint_path} does not contain a 'model' entry")

    model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
    yolo = YOLO(model_cfg)
    model_obj = checkpoint['model']

    example_input = torch.randn(1, 3, imgsz, imgsz)

    if isinstance(model_obj, dict):
        LOGGER.info("   Checkpoint stores QAT state_dict; rebuilding model graph...")
        yolo.model = yolo.model.prepare_for_qat(
            backend=backend,
            example_input=example_input,
        )
        yolo.model.load_state_dict(model_obj, strict=False)
    else:
        LOGGER.info("   Checkpoint provides full QAT model object; loading directly...")
        yolo.model = model_obj

    ensure_module_bookkeeping(yolo.model, recursive=True)
    yolo.model.eval()
    
    # Convert model to float32 if it's in half precision (required for quantization)
    LOGGER.info("   Converting model to float32 (required for quantization)...")
    yolo.model = yolo.model.float()

    LOGGER.info("   Converting QAT model to INT8...")
    converted = yolo.model.convert_to_quantized()
    if converted is not None:
        yolo.model = converted

    ensure_module_bookkeeping(yolo.model, recursive=True)

    fakequants_remaining = sum(
        1 for module in yolo.model.modules()
        if module.__class__.__name__.lower().startswith('fakequant')
    )
    if fakequants_remaining > 0:
        LOGGER.warning(f"   ⚠️  Detected {fakequants_remaining} FakeQuant modules after conversion.")
    else:
        LOGGER.info("   ✓ QAT observers successfully removed (pure INT8 graph)")

    int8_checkpoint = {
        'model': yolo.model,
        'backend': backend,
        'source_qat': str(qat_path),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(int8_checkpoint, output_path)
    LOGGER.info(f"   ✓ Saved INT8 checkpoint to {output_path}")

    return str(output_path)


def evaluate_int8_model(
    int8_checkpoint_path='runs/detect/train/weights/best_int8.pt',
    fp32_checkpoint_path=None,  # Optional: for comparison
    data_cfg='obc-yolov8/ultralytics10.24/ultralytics/cfg/datasets/combined_china_motorbike.yaml',
    imgsz=640,
    device=0,
    batch=16,
    save_fixed_checkpoint=True
):
    """
    Evaluate INT8 quantized model and optionally compare with FP32 model.
    
    Args:
        int8_checkpoint_path: Path to INT8 model checkpoint
        fp32_checkpoint_path: Optional path to FP32 model for comparison
        data_cfg: Dataset configuration file
        imgsz: Image size
        device: Device to use (0 for GPU, 'cpu' for CPU)
        batch: Batch size for evaluation
        save_fixed_checkpoint: If True, write a repaired checkpoint with restored module bookkeeping
    """
    LOGGER.info("=" * 80)
    LOGGER.info("Evaluating INT8 Quantized Model")
    LOGGER.info("=" * 80)
    
    # Check if INT8 checkpoint exists
    int8_path = Path(int8_checkpoint_path)
    if not int8_path.exists():
        LOGGER.error(f"INT8 checkpoint not found: {int8_checkpoint_path}")
        LOGGER.info("Available checkpoints:")
        checkpoint_dir = int8_path.parent
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.pt"):
                LOGGER.info(f"  - {f}")
        return None
    
    # Load INT8 model
    LOGGER.info(f"\n[1/3] Loading INT8 model from {int8_checkpoint_path}...")
    try:
        # CRITICAL: First peek at checkpoint to get backend, then set engine BEFORE loading
        # This must be done before any quantized operations
        checkpoint_metadata = torch.load(int8_checkpoint_path, map_location='cpu', weights_only=False)
        backend = checkpoint_metadata.get('backend', 'fbgemm')
        LOGGER.info(f"   Backend: {backend}")
        
        # CRITICAL: Set quantization backend engine IMMEDIATELY
        # This must be set before any quantized modules are created or used
        if backend == 'fbgemm':
            torch.backends.quantized.engine = 'fbgemm'
            LOGGER.info("   ✓ Set quantization backend engine to fbgemm (must be before model operations)")
        elif backend == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'
            LOGGER.info("   ✓ Set quantization backend engine to qnnpack")
        
        # Now load the full checkpoint
        int8_checkpoint = checkpoint_metadata
        
        if 'model' in int8_checkpoint:
            model_obj = int8_checkpoint['model']
            
            # Check if it's a full model object or state_dict
            if isinstance(model_obj, dict):
                # It's a state_dict - need to load model structure first
                LOGGER.info("   Checkpoint contains state_dict, loading model structure...")
                
                # Load model structure from config
                model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
                model = YOLO(model_cfg)
                
                # CRITICAL: For INT8 models, we need to prepare QAT then convert BEFORE loading state_dict
                # This ensures the model structure matches the saved INT8 state_dict
                LOGGER.info("   Preparing model structure and converting to INT8...")
                example_input = torch.randn(1, 3, imgsz, imgsz)
                
                # Prepare for QAT (backend engine already set above)
                model.model = model.model.prepare_for_qat(
                    backend=backend,  # Use the backend from checkpoint
                    example_input=example_input
                )
                
                # Convert to INT8 (this replaces FakeQuantize with QuantizedConv2d, etc.)
                model.model = model.model.convert_to_quantized()
                
                # Now load the INT8 state_dict
                LOGGER.info("   Loading INT8 weights from state_dict...")
                model.model.load_state_dict(model_obj, strict=False)
                
                # CRITICAL: Clean up any remaining FakeQuantize modules that might cause issues
                # Some FakeQuantize modules might remain after conversion if they were disabled
                from torch.ao.quantization import FakeQuantize
                import torch.nn as nn
                
                # Remove any FakeQuantize modules that don't have activation_post_process
                # (these are invalid and cause errors during repr)
                cleanup_count = 0
                for name, module in list(model.model.named_modules()):
                    if isinstance(module, FakeQuantize):
                        if not hasattr(module, 'activation_post_process'):
                            # This is an invalid FakeQuantize - remove it
                            parent_name = '.'.join(name.split('.')[:-1])
                            child_name = name.split('.')[-1]
                            if parent_name:
                                try:
                                    parent = dict(model.model.named_modules())[parent_name]
                                    # Replace with identity module
                                    setattr(parent, child_name, nn.Identity())
                                    cleanup_count += 1
                                except:
                                    pass
                
                if cleanup_count > 0:
                    LOGGER.info(f"   Cleaned up {cleanup_count} invalid FakeQuantize modules")
                
                LOGGER.info("   ✓ INT8 model loaded from state_dict")
                
            else:
                # It's a full model object - much easier!
                LOGGER.info("   Checkpoint contains full model object, loading directly...")
                
                # Create YOLO wrapper and set the model
                model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
                model = YOLO(model_cfg)
                model.model = model_obj
                
                # Skip cleanup for quantized models - they don't support named_modules() iteration
                # QuantizedConv2d modules have different structure
                LOGGER.info("   ✓ INT8 model loaded successfully (quantized model structure)")
                
        elif 'model_state_dict' in int8_checkpoint:
            # Fallback: use model_state_dict key
            LOGGER.info("   Loading from model_state_dict...")
            model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
            model = YOLO(model_cfg)
            example_input = torch.randn(1, 3, imgsz, imgsz)
            model.model = model.model.prepare_for_qat(
                backend=backend,  # Use the backend from checkpoint
                example_input=example_input
            )
            model.model = model.model.convert_to_quantized()
            model.model.load_state_dict(int8_checkpoint['model_state_dict'], strict=False)
            
            # Clean up invalid FakeQuantize modules (with error handling)
            from torch.ao.quantization import FakeQuantize
            import torch.nn as nn
            cleanup_count = 0
            try:
                modules_to_check = []
                try:
                    modules_to_check = list(model.model.named_modules())
                except AttributeError:
                    modules_to_check = []
                
                for name, module in modules_to_check:
                    if isinstance(module, FakeQuantize):
                        if not hasattr(module, 'activation_post_process'):
                            parent_name = '.'.join(name.split('.')[:-1])
                            child_name = name.split('.')[-1]
                            if parent_name:
                                try:
                                    parent = dict(model.model.named_modules())[parent_name]
                                    setattr(parent, child_name, nn.Identity())
                                    cleanup_count += 1
                                except:
                                    pass
                if cleanup_count > 0:
                    LOGGER.info(f"   Cleaned up {cleanup_count} invalid FakeQuantize modules")
            except Exception as cleanup_error:
                LOGGER.warning(f"   ⚠️  Cleanup failed: {cleanup_error}, continuing anyway")
            
            LOGGER.info("   ✓ INT8 model loaded from model_state_dict")
            
        else:
            LOGGER.error("   ✗ Checkpoint format not recognized")
            LOGGER.error(f"   Available keys: {list(int8_checkpoint.keys())}")
            return None
            
    except Exception as e:
        LOGGER.error(f"   ✗ Failed to load INT8 model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    bookkeeping_repairs = _repair_quantized_bookkeeping(getattr(model, 'model', None))
    if bookkeeping_repairs > 0:
        LOGGER.info(f"   ✓ Repaired missing bookkeeping on {bookkeeping_repairs} modules")
        if save_fixed_checkpoint:
            fixed_path = int8_path.with_name(f"{int8_path.stem}_bookkeeping{int8_path.suffix}")
            try:
                if isinstance(int8_checkpoint, dict):
                    int8_checkpoint['model'] = getattr(model, 'model', None)
                    int8_checkpoint['bookkeeping_fix_applied'] = True
                    int8_checkpoint['bookkeeping_fix_modules'] = bookkeeping_repairs
                torch.save(int8_checkpoint, fixed_path)
                LOGGER.info(f"   ✓ Saved repaired checkpoint to {fixed_path}")
            except Exception as save_error:
                LOGGER.warning(f"   ⚠️ Failed to save repaired checkpoint ({save_error})")
    else:
        LOGGER.info("   Bookkeeping already intact on quantized modules")
    
    # Evaluate INT8 model
    LOGGER.info(f"\n[2/3] Evaluating INT8 model...")
    LOGGER.info(f"   Dataset: {data_cfg}")
    LOGGER.info(f"   Image size: {imgsz}")
    LOGGER.info(f"   Batch size: {batch}")
    
    # CRITICAL: fbgemm and qnnpack backend quantized models MUST run on CPU
    # Both are CPU-only quantization backends
    # Backend engine was already set during model loading
    if backend in ['fbgemm', 'qnnpack']:
        LOGGER.info(f"   ⚠️  {backend} backend detected - quantized operations require CPU")
        eval_device = 'cpu'
        
        # Try to move to CPU, but quantized models may not support .cpu()
        # Quantized models use _packed_params and don't support standard PyTorch operations
        try:
            # Try to check device - quantized models may not have standard parameters
            try:
                # Check if any parameters exist and their device
                has_params = False
                model_device = None
                for param in model.model.parameters(recurse=False):
                    has_params = True
                    model_device = param.device
                    break
                
                # If model has parameters and they're not on CPU, try to move
                if has_params and model_device != torch.device('cpu'):
                    LOGGER.info("   Moving model to CPU...")
                    model.model = model.model.cpu()
                else:
                    LOGGER.info("   Model already on CPU (or quantized - no explicit move needed)")
            except (StopIteration, AttributeError):
                # No standard parameters found - likely quantized model
                LOGGER.info("   Quantized model detected (no standard parameters)")
                LOGGER.info("   Skipping .cpu() call (quantized models are CPU-only)")
        except (AttributeError, RuntimeError, TypeError) as e:
            # Quantized models don't support .cpu() or parameter iteration
            # This is expected - quantized models are already CPU-only
            LOGGER.info(f"   Quantized model structure detected: {type(e).__name__}")
            LOGGER.info("   Continuing with CPU evaluation (quantized models are CPU-only by design)")
    else:
        eval_device = device
    
    # Ensure model is in eval mode for evaluation
    # Quantized models may not support .eval() due to QuantizedConv2d structure
    # They're already in eval mode after conversion, so we can skip this
    is_quantized = _is_quantized_model(model.model)
    if is_quantized:
        LOGGER.info("   Quantized model detected - skipping .eval() (already in eval mode)")
    else:
        try:
            model.model.eval()
            LOGGER.info("   ✓ Model set to eval mode")
        except (AttributeError, RuntimeError) as e:
            # Fallback: if .eval() fails, assume it's quantized
            LOGGER.info("   Model does not support .eval() - assuming quantized (already in eval mode)")
    
    # Try INT8 evaluation first
    # NOTE: Quantized models may have structural issues that prevent forward pass
    # If forward pass fails, we'll fall back to QAT model evaluation
    int8_results = None
    used_qat_fallback = False
    
    # Sanitize: convert any lingering QAT modules to float before forward
    def _sanitize_qat_modules(root_module):
        try:
            # First, specifically handle ODConv's internal QAT modules
            # ODConv must remain FP32, so convert any QAT modules inside it
            try:
                from ultralytics.nn.ODConv import ODConv
                from ultralytics.nn.BoTNet import BoTNet
                from ultralytics.nn.CA_Attention import CoordAtt
                from ultralytics.nn.modules.block import DFL
                all_modules_dict = dict(root_module.named_modules())
                odconv_converted = 0
                
                target_parent_types = (ODConv, BoTNet, CoordAtt, DFL)

                for full_name, module in all_modules_dict.items():
                    if isinstance(module, target_parent_types):
                        # ODConv is a Sequential, so we need to check its children
                        # Also check nested modules like Attention
                        for submodule_name, submodule in module.named_modules():
                            if submodule_name == '':
                                continue
                            
                            # Check if this submodule is a QAT module
                            # Also check if it's from QAT namespace
                            mod_ns = getattr(type(submodule), '__module__', '')
                            is_qat_ns = 'torch.ao.nn.qat' in mod_ns or 'qat' in mod_ns
                            has_wfq = hasattr(submodule, 'weight_fake_quant')
                            has_to_float = hasattr(submodule, 'to_float')
                            is_qat_module = (has_wfq and has_to_float) or is_qat_ns
                            
                            if is_qat_module and has_to_float:
                                try:
                                    # Convert QAT module to FP32
                                    float_mod = submodule.to_float()
                                    
                                    # Construct the full path and replace
                                    if submodule_name:
                                        full_submodule_path = f"{full_name}.{submodule_name}"
                                    else:
                                        full_submodule_path = full_name
                                    
                                    # Navigate to parent and replace
                                    # Handle both attribute access and Sequential indexing
                                    path_parts = full_submodule_path.split('.')
                                    
                                    # Navigate through the path to find the parent
                                    current = root_module
                                    try:
                                        # Navigate through all parts except the last one
                                        for part in path_parts[:-1]:
                                            if hasattr(current, part):
                                                current = getattr(current, part)
                                            elif isinstance(current, (torch.nn.Sequential, torch.nn.ModuleList)):
                                                try:
                                                    idx = int(part)
                                                    if idx < len(current):
                                                        current = current[idx]
                                                    else:
                                                        break
                                                except (ValueError, IndexError, TypeError):
                                                    break
                                            else:
                                                break
                                        else:
                                            # Successfully navigated to parent
                                            child_name = path_parts[-1]
                                            # Try attribute access first
                                            if hasattr(current, child_name):
                                                setattr(current, child_name, float_mod)
                                                odconv_converted += 1
                                            # Try Sequential/ModuleList indexing
                                            elif isinstance(current, (torch.nn.Sequential, torch.nn.ModuleList)):
                                                try:
                                                    idx = int(child_name)
                                                    if idx < len(current):
                                                        current[idx] = float_mod
                                                        odconv_converted += 1
                                                except (ValueError, IndexError, TypeError):
                                                    pass
                                    except Exception:
                                        pass
                                except Exception as e:
                                    pass
                
                if odconv_converted > 0:
                    LOGGER.info(f"   ✓ Converted {odconv_converted} QAT modules inside ODConv/BoTNet/CoordAtt/DFL to FP32 (runtime fix)")
            except Exception as e:
                pass
            
            # Walk modules deepest-first so replacements don't break traversal
            mods = list(root_module.named_modules())
            for name, module in sorted(mods, key=lambda x: len(x[0]), reverse=True):
                mod_ns = getattr(type(module), '__module__', '')
                is_qat_ns = 'torch.ao.nn.qat' in mod_ns or 'qat' in mod_ns
                can_to_float = hasattr(module, 'to_float')
                has_wfq = hasattr(module, 'weight_fake_quant')
                # Convert any QAT module (namespace match) or anything with to_float()/weight_fake_quant
                if can_to_float and (is_qat_ns or has_wfq):
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1] if name else ''
                    try:
                        float_mod = module.to_float()
                        if parent_name:
                            parent = dict(root_module.named_modules()).get(parent_name)
                            if parent is not None and hasattr(parent, child_name):
                                setattr(parent, child_name, float_mod)
                    except Exception:
                        pass
        except Exception:
            pass

    _sanitize_qat_modules(model.model)

    # Test if model can do a forward pass first
    try:
        LOGGER.info("   Testing model forward pass capability...")
        test_input = torch.randn(1, 3, imgsz, imgsz)
        
        # For quantized models, check if quant/dequant stubs exist
        has_quant_stub = hasattr(model.model, 'quant') and isinstance(model.model.quant, torch.ao.quantization.QuantStub)
        has_dequant_stub = hasattr(model.model, 'dequant') and isinstance(model.model.dequant, torch.ao.quantization.DeQuantStub)
        
        with torch.no_grad():
            if has_quant_stub and has_dequant_stub:
                # PTQ model with quant/dequant stubs - use them properly
                LOGGER.info("   Using QuantStub/DeQuantStub for quantized model...")
                x_quant = model.model.quant(test_input)
                output = model.model(x_quant)
                _ = model.model.dequant(output)
            else:
                # Try direct forward (might work for some quantized models)
                _ = model.model(test_input)
        
        LOGGER.info("   ✓ Forward pass test succeeded")
        forward_pass_works = True
    except (AttributeError, RuntimeError, TypeError, NotImplementedError) as e:
        # Print full error for debugging
        import traceback
        error_msg = str(e)
        error_type = type(e).__name__
        LOGGER.warning(f"   ⚠️  Forward pass test failed: {error_type}: {error_msg}")
        
        # Print traceback to see exactly where it fails
        LOGGER.info("   Full traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                LOGGER.info(f"   {line}")
        
        if "'Conv2d' object has no attribute" in error_msg or "'backward_hooks'" in error_msg or "'_modules'" in error_msg:
            LOGGER.warning("   Model structure incompatible with quantized Conv2d modules")
            LOGGER.info("   Will skip INT8 evaluation and use QAT fallback instead")
            forward_pass_works = False
        elif "method" in error_msg.lower() or "callable" in error_msg.lower():
            LOGGER.warning("   Weight access issue - may be from QuantizedConv or other quantized modules")
            LOGGER.info("   Will skip INT8 evaluation and use QAT fallback instead")
            forward_pass_works = False
        elif "quantized::" in error_msg or "QuantizedCPU" in error_msg or "NotImplementedError" in error_type:
            # Quantized operation backend issue - this is a known PyTorch limitation
            # The quantized Conv2d operations require QuantizedCPU backend but tensors are on CPU backend
            LOGGER.warning("   ⚠️  Quantized operations backend mismatch detected")
            LOGGER.warning("   This is a known issue with PyTorch quantized operations:")
            LOGGER.warning("   - Quantized Conv2d requires QuantizedCPU backend")
            LOGGER.warning("   - But intermediate tensors may be on regular CPU backend")
            LOGGER.warning("   - This can happen even with CPU-only PyTorch builds")
            LOGGER.warning("   ")
            LOGGER.warning("   Possible causes:")
            LOGGER.warning("   1. Model conversion issue - quantized ops not properly set up")
            LOGGER.warning("   2. PyTorch version bug with quantized backend dispatch")
            LOGGER.warning("   3. Model saved/loaded incorrectly, losing backend context")
            LOGGER.warning("   ")
            LOGGER.warning("   Workaround: Skip forward pass test, try evaluation directly")
            LOGGER.warning("   (Evaluation uses model.val() which may handle quantization differently)")
            forward_pass_works = True  # Try evaluation anyway - model.val() might work
        else:
            # Different error - might still work for evaluation
            LOGGER.warning(f"   Forward pass test failed with unexpected error")
            forward_pass_works = True  # Try anyway
    
    if not forward_pass_works:
        # Forward pass test failed - skip INT8 evaluation and go straight to QAT fallback
        LOGGER.info("\n   Skipping INT8 evaluation (model structure incompatible)")
        int8_results = None
    else:
        # Even if forward pass test failed, try evaluation - model.val() might use different code path
        LOGGER.info("\n   Attempting INT8 model evaluation (forward pass test was skipped/failed)...")
        LOGGER.info("   Note: If this fails, the model may need to be re-converted with PTQ")
        try:
            int8_results = model.val(
                data=data_cfg,
                imgsz=imgsz,
                batch=batch,
                device=eval_device,
                plots=False,
                save=False,
                verbose=True
            )
        except (NotImplementedError, RuntimeError, AttributeError) as e:
            error_str = str(e)
            if ('quantized::conv2d' in error_str or 'QuantizedCPU' in error_str or 
                'backend' in error_str.lower() or 
                "'Conv2d' object has no attribute" in error_str or
                "'backward_hooks'" in error_str or "'_modules'" in error_str):
                # Backend issue with INT8 model - fall back to QAT model
                LOGGER.warning(f"\n   ⚠️  INT8 model evaluation failed due to backend limitation:")
                LOGGER.warning(f"   {str(e)[:200]}...")
                int8_results = None
            else:
                # Different error - re-raise
                raise
    
    # If INT8 evaluation didn't work, try QAT fallback
    if int8_results is None:
        LOGGER.info("\n   Falling back to QAT model (FakeQuantize) for evaluation...")
        LOGGER.info("   Note: QAT model provides accuracy metrics (simulated quantization)")
        
        # Try to load QAT checkpoint
        qat_checkpoint_path = Path('runs/detect/train/weights/best_qat.pt')
        if qat_checkpoint_path.exists():
            LOGGER.info(f"   Loading QAT model from {qat_checkpoint_path}...")
            try:
                qat_checkpoint = torch.load(qat_checkpoint_path, map_location='cpu', weights_only=False)
                
                if 'model' in qat_checkpoint:
                    qat_model_obj = qat_checkpoint['model']
                    
                    if isinstance(qat_model_obj, dict):
                        # Load QAT model structure
                        model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
                        model = YOLO(model_cfg)
                        model.model = model.model.prepare_for_qat(
                            backend=backend,
                            example_input=torch.randn(1, 3, imgsz, imgsz)
                        )
                        model.model.load_state_dict(qat_model_obj, strict=False)
                    else:
                        # Full model object
                        model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
                        model = YOLO(model_cfg)
                        model.model = qat_model_obj
                    
                    model.model.eval()
                    LOGGER.info("   ✓ QAT model loaded successfully")
                    
                    # Evaluate QAT model (should work fine)
                    int8_results = model.val(
                        data=data_cfg,
                        imgsz=imgsz,
                        batch=batch,
                        device=eval_device if backend != 'fbgemm' else device,  # QAT can use GPU
                        plots=False,
                        save=False,
                        verbose=True
                    )
                    used_qat_fallback = True
                    LOGGER.info("   ✓ QAT model evaluation completed")
                else:
                    LOGGER.error("   ✗ QAT checkpoint format not recognized")
                    int8_results = None
            except Exception as qat_e:
                LOGGER.error(f"   ✗ Failed to load/evaluate QAT model: {qat_e}")
                import traceback
                traceback.print_exc()
                int8_results = None
        else:
            LOGGER.error(f"   ✗ QAT checkpoint not found at {qat_checkpoint_path}")
            LOGGER.error("   Cannot fall back to QAT model evaluation")
            int8_results = None
    
    # Extract metrics (if evaluation succeeded)
    if int8_results is not None and hasattr(int8_results, 'box'):
        box = int8_results.box
        model_type = "QAT Model (FakeQuantize)" if used_qat_fallback else "INT8 Model (Quantized)"
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info(f"{model_type} Metrics:")
        LOGGER.info("=" * 80)
        if used_qat_fallback:
            LOGGER.info("  Note: Using QAT model due to INT8 backend limitation")
            LOGGER.info("  Metrics reflect simulated quantization (FakeQuantize)")
        LOGGER.info(f"  mAP@0.5:      {box.map50:.4f} ({box.map50*100:.2f}%)")
        LOGGER.info(f"  mAP@0.5:0.95: {box.map:.4f} ({box.map*100:.2f}%)")
        LOGGER.info(f"  Precision:    {box.mp:.4f} ({box.mp*100:.2f}%)")
        LOGGER.info(f"  Recall:        {box.mr:.4f} ({box.mr*100:.2f}%)")
        
        # F1-score
        f1 = 2 * (box.mp * box.mr) / (box.mp + box.mr) if (box.mp + box.mr) > 0 else 0.0
        LOGGER.info(f"  F1-Score:      {f1:.4f} ({f1*100:.2f}%)")
        
        # Speed metrics
        if hasattr(int8_results, 'speed'):
            speed = int8_results.speed
            LOGGER.info(f"\n  Inference time: {speed.get('inference', 'N/A'):.2f} ms/img")
            LOGGER.info(f"  Preprocessing:  {speed.get('preprocess', 'N/A'):.2f} ms/img")
            LOGGER.info(f"  Postprocessing: {speed.get('postprocess', 'N/A'):.2f} ms/img")
        
        int8_metrics = {
            'map50': box.map50,
            'map': box.map,
            'precision': box.mp,
            'recall': box.mr,
            'f1': f1,
            'speed': int8_results.speed if hasattr(int8_results, 'speed') else None
        }
    else:
        LOGGER.warning("   ⚠️  Could not extract box metrics from results")
        int8_metrics = None
    
    # Optional: Compare with FP32 model
    if fp32_checkpoint_path and Path(fp32_checkpoint_path).exists():
        LOGGER.info(f"\n[3/3] Evaluating FP32 model for comparison...")
        try:
            fp32_model = YOLO(fp32_checkpoint_path)
            fp32_results = fp32_model.val(
                data=data_cfg,
                imgsz=imgsz,
                batch=batch,
                device=device,
                plots=False,
                save=False,
                verbose=False
            )
            
            if hasattr(fp32_results, 'box'):
                fp32_box = fp32_results.box
                LOGGER.info("\n" + "=" * 80)
                LOGGER.info("FP32 Model Metrics (for comparison):")
                LOGGER.info("=" * 80)
                LOGGER.info(f"  mAP@0.5:      {fp32_box.map50:.4f} ({fp32_box.map50*100:.2f}%)")
                LOGGER.info(f"  mAP@0.5:0.95: {fp32_box.map:.4f} ({fp32_box.map*100:.2f}%)")
                LOGGER.info(f"  Precision:    {fp32_box.mp:.4f} ({fp32_box.mp*100:.2f}%)")
                LOGGER.info(f"  Recall:        {fp32_box.mr:.4f} ({fp32_box.mr*100:.2f}%)")
                
                fp32_f1 = 2 * (fp32_box.mp * fp32_box.mr) / (fp32_box.mp + fp32_box.mr) if (fp32_box.mp + fp32_box.mr) > 0 else 0.0
                LOGGER.info(f"  F1-Score:      {fp32_f1:.4f} ({fp32_f1*100:.2f}%)")
                
                # Comparison
                if int8_metrics:
                    LOGGER.info("\n" + "=" * 80)
                    LOGGER.info("Comparison (INT8 vs FP32):")
                    LOGGER.info("=" * 80)
                    
                    map50_diff = int8_metrics['map50'] - fp32_box.map50
                    map_diff = int8_metrics['map'] - fp32_box.map
                    prec_diff = int8_metrics['precision'] - fp32_box.mp
                    recall_diff = int8_metrics['recall'] - fp32_box.mr
                    
                    LOGGER.info(f"  mAP@0.5:      {map50_diff:+.4f} ({map50_diff*100:+.2f}%)")
                    LOGGER.info(f"  mAP@0.5:0.95: {map_diff:+.4f} ({map_diff*100:+.2f}%)")
                    LOGGER.info(f"  Precision:    {prec_diff:+.4f} ({prec_diff*100:+.2f}%)")
                    LOGGER.info(f"  Recall:       {recall_diff:+.4f} ({recall_diff*100:+.2f}%)")
                    
                    # Calculate accuracy retention
                    if fp32_box.map50 > 0:
                        retention = (int8_metrics['map50'] / fp32_box.map50) * 100
                        LOGGER.info(f"\n  Accuracy retention: {retention:.2f}%")
                    
                    # Speed comparison if available
                    if hasattr(fp32_results, 'speed') and int8_metrics.get('speed'):
                        fp32_speed = fp32_results.speed.get('inference', 0)
                        int8_speed = int8_metrics['speed'].get('inference', 0)
                        if fp32_speed > 0 and int8_speed > 0:
                            speedup = fp32_speed / int8_speed
                            LOGGER.info(f"  Speedup: {speedup:.2f}x")
                    
        except Exception as e:
            LOGGER.warning(f"   ⚠️  Could not evaluate FP32 model: {e}")
    
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Evaluation Complete!")
    LOGGER.info("=" * 80)
    
    return int8_metrics


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate INT8 quantized YOLOv8-CA model')
    parser.add_argument('--int8', type=str, default='runs/detect/train/weights/best_int8.pt',
                       help='Path to INT8 model checkpoint')
    parser.add_argument('--fp32', type=str, default=None,
                       help='Optional: Path to FP32 model for comparison')
    parser.add_argument('--data', type=str, 
                       default='obc-yolov8/ultralytics10.24/ultralytics/cfg/datasets/combined_china_motorbike.yaml',
                       help='Dataset configuration file')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    def device_type(value):
        """Convert device string to int or keep as 'cpu'."""
        if value.lower() == 'cpu':
            return 'cpu'
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Device must be 'cpu' or an integer, got: {value}")
    
    parser.add_argument('--device', type=device_type, default=0, help='Device (0 for GPU, "cpu" for CPU)')
    parser.add_argument('--no-save-fixed', action='store_true', help='Disable writing repaired INT8 checkpoint')
    parser.add_argument('--qat-checkpoint', type=str, default=None,
                        help='Optional QAT checkpoint to convert to INT8 before evaluation')
    parser.add_argument('--export-int8', action='store_true',
                        help='Convert the provided QAT checkpoint to INT8 and persist it')
    parser.add_argument('--out', type=str, default=None,
                        help='Output path for exported INT8 checkpoint (defaults to <qat>_int8.pt)')
    
    args = parser.parse_args()
    
    int8_path = args.int8

    if args.qat_checkpoint:
        export_path = Path(args.out) if args.out else Path(args.qat_checkpoint).with_name(f"{Path(args.qat_checkpoint).stem}_int8.pt")
        try:
            int8_path = convert_qat_checkpoint(
                qat_checkpoint_path=args.qat_checkpoint,
                output_path=str(export_path),
                imgsz=args.imgsz,
            )
            if args.export_int8:
                LOGGER.info(f"INT8 export path: {int8_path}")
        except Exception as conversion_error:
            LOGGER.error(f"Failed to convert QAT checkpoint '{args.qat_checkpoint}': {conversion_error}")
            return
    
    evaluate_int8_model(
        int8_checkpoint_path=int8_path,
        fp32_checkpoint_path=args.fp32,
        data_cfg=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_fixed_checkpoint=not args.no_save_fixed
    )


if __name__ == '__main__':
    main()

