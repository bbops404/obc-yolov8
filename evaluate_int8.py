"""
Evaluate INT8 quantized YOLOv8-CA model
Compares performance metrics between FP32 and INT8 models
"""

import sys
import time
import argparse
import traceback
from pathlib import Path
import warnings
import numpy as np
import torch

# --- PATH SETUP ---
# Robustly add local ultralytics to path
# We look for the folder relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
ULTRALYTICS_PATH = SCRIPT_DIR / 'obc-yolov8' / 'ultralytics10.24'

if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))
else:
    print(f"WARNING: Local Ultralytics path not found at {ULTRALYTICS_PATH}")
    print("Attempting to use system-installed ultralytics (may cause version mismatches)...")

from ultralytics import YOLO
from ultralytics.nn.tasks import ensure_module_bookkeeping
from ultralytics.utils import LOGGER

warnings.filterwarnings('ignore')

# --- QUANTIZATION BACKEND SETUP ---
# CRITICAL: Set default quantization backend early
# This must be set before any quantized operations are created
try:
    # Default to fbgemm (x86) or qnnpack (ARM) based on system availability
    if 'fbgemm' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'fbgemm'
    elif 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
except Exception as e:
    LOGGER.warning(f"Could not set quantized engine: {e}")


def _is_quantized_model(model):
    """
    Check if a model is quantized by looking for QuantizedConv2d or _packed_params.
    """
    try:
        if hasattr(model, 'model'):
            model = model.model
            
        # Try to iterate modules - quantized models may fail here if bookkeeping is broken
        try:
            for _, module in model.named_modules():
                module_type = type(module).__name__
                if 'Quantized' in module_type:
                    return True
                if hasattr(module, '_packed_params'):
                    return True
        except (AttributeError, RuntimeError):
            # If named_modules fails, it's almost certainly a broken quantized model
            return True
            
        return False
    except Exception:
        return False


def _repair_quantized_bookkeeping(root_module):
    """Repair missing nn.Module bookkeeping attributes on quantized modules."""
    if root_module is None:
        return 0

    visited = set()
    stack = [root_module]
    modules_needing_fix = 0

    while stack:
        module = stack.pop()
        if not hasattr(module, '_modules'): # Basic sanity check
            continue
            
        module_id = id(module)
        if module_id in visited:
            continue
        visited.add(module_id)

        def _is_dict_like(value):
            return isinstance(value, dict)

        needs_fix = False
        
        # Check standard PyTorch internal attributes
        internal_attrs = ['_modules', '_parameters', '_buffers']
        for attr in internal_attrs:
            try:
                if not _is_dict_like(getattr(module, attr, None)):
                    needs_fix = True
            except Exception:
                needs_fix = True

        # CRITICAL: Fix hook attributes
        hook_attrs = ('_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_backward_pre_hooks',
                     '_state_dict_hooks', '_load_state_dict_pre_hooks')
        
        for hook_attr in hook_attrs:
            try:
                attr_value = getattr(module, hook_attr, None)
                if not _is_dict_like(attr_value):
                    object.__setattr__(module, hook_attr, {})
                    needs_fix = True
            except Exception:
                # Force create
                object.__setattr__(module, hook_attr, {})
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
            elif hasattr(module, 'children'):
                stack.extend(list(module.children()))
        except Exception:
            pass

    try:
        ensure_module_bookkeeping(root_module, recursive=True)
    except Exception:
        pass
        
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

    checkpoint = torch.load(qat_path, map_location='cpu')
    backend = checkpoint.get('backend', 'fbgemm')
    
    # Set engine for conversion
    if backend in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = backend
    
    # Structure definition
    model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
    yolo = YOLO(model_cfg)
    
    # Load weights
    if 'model' in checkpoint:
        model_obj = checkpoint['model']
        if isinstance(model_obj, dict):
            # State dict approach (most robust for QAT)
            LOGGER.info("   Checkpoint stores QAT state_dict; rebuilding model graph...")
            example_input = torch.randn(1, 3, imgsz, imgsz)
            yolo.model = yolo.model.prepare_for_qat(backend=backend, example_input=example_input)
            yolo.model.load_state_dict(model_obj, strict=False)
        else:
            # Full object approach
            LOGGER.info("   Checkpoint provides full QAT model object...")
            yolo.model = model_obj
    else:
        raise KeyError("Checkpoint must contain 'model' key")

    # Ensure float32 before conversion
    yolo.model = yolo.model.float().cpu()
    yolo.model.eval()

    LOGGER.info("   Converting QAT model to INT8...")
    # Convert performs the actual fusion and quantization
    converted_model = yolo.model.convert_to_quantized()
    if converted_model is not None:
        yolo.model = converted_model

    # Fix bookkeeping immediately after conversion
    _repair_quantized_bookkeeping(yolo.model)

    int8_checkpoint = {
        'model': yolo.model,
        'backend': backend,
        'source_qat': str(qat_path),
        'imgsz': imgsz
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(int8_checkpoint, output_path)
    LOGGER.info(f"   ✓ Saved INT8 checkpoint to {output_path}")

    return str(output_path)


def evaluate_int8_model(
    int8_checkpoint_path='runs/detect/train/weights/best_int8.pt',
    fp32_checkpoint_path=None,
    data_cfg=None,
    imgsz=640,
    device=0,
    batch=16,
    save_fixed_checkpoint=True,
    skip_evaluation=False
):
    LOGGER.info("=" * 80)
    LOGGER.info("Evaluating INT8 Quantized Model")
    LOGGER.info("=" * 80)
    
    int8_path = Path(int8_checkpoint_path)
    if not int8_path.exists():
        LOGGER.error(f"INT8 checkpoint not found: {int8_path}")
        return None
    
    # --- LOAD INT8 MODEL ---
    LOGGER.info(f"\n[1/3] Loading INT8 model from {int8_checkpoint_path}...")
    try:
        # Load meta first to get backend
        chk_meta = torch.load(int8_path, map_location='cpu', weights_only=False)
        backend = chk_meta.get('backend', 'fbgemm')
        LOGGER.info(f"   Backend: {backend}")
        
        if backend in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = backend
        
        # Load Model
        if 'model' in chk_meta and not isinstance(chk_meta['model'], dict):
            # Full object load (Preferred for INT8)
            yolo_int8 = YOLO('yolov8n.yaml') # Dummy init
            yolo_int8.model = chk_meta['model']
        else:
            # State Dict load (Complex path)
            LOGGER.info("   Loading from state_dict (rebuilding graph)...")
            model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
            yolo_int8 = YOLO(model_cfg)
            
            # Rebuild structure
            example_input = torch.randn(1, 3, imgsz, imgsz)
            yolo_int8.model = yolo_int8.model.prepare_for_qat(backend=backend, example_input=example_input)
            yolo_int8.model = yolo_int8.model.convert_to_quantized()
            
            # Load weights
            state_dict = chk_meta.get('model', chk_meta.get('model_state_dict'))
            yolo_int8.model.load_state_dict(state_dict, strict=False)
            
        LOGGER.info("   ✓ INT8 model loaded")
        
    except Exception as e:
        LOGGER.error(f"   ✗ Failed to load INT8 model: {e}")
        traceback.print_exc()
        return None

    # Repair Bookkeeping
    repairs = _repair_quantized_bookkeeping(yolo_int8.model)
    if repairs > 0:
        LOGGER.info(f"   ✓ Repaired bookkeeping on {repairs} modules")
        if save_fixed_checkpoint:
            try:
                chk_meta['model'] = yolo_int8.model
                torch.save(chk_meta, int8_path.with_name(f"{int8_path.stem}_fixed.pt"))
            except: pass

    # --- DEVICE HANDLING ---
    # Quantized models (fbgemm/qnnpack) MUST run on CPU.
    # We ignore the 'device' argument for the INT8 model.
    int8_device = 'cpu'
    yolo_int8.model.to('cpu')
    LOGGER.info("   ℹ️  Forcing INT8 model to CPU (required for quantization backend)")

    # --- EVALUATION ---
    int8_metrics = None
    int8_results = None

    if not skip_evaluation:
        LOGGER.info(f"\n[2/3] Evaluating INT8 model on {data_cfg}...")
        try:
            # FIXED: half=False is mandatory for INT8. 
            # Default ultralytics val() uses half=True which crashes quantized models.
            int8_results = yolo_int8.val(
                data=data_cfg,
                imgsz=imgsz,
                batch=batch,
                device=int8_device, # Force CPU
                plots=False,
                save=False,
                verbose=True,
                half=False  # CRITICAL FIX
            )
            
            if hasattr(int8_results, "box"):
                box = int8_results.box

                # Compute F1-score from precision and recall
                prec = box.mp
                rec = box.mr
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                int8_metrics = {
                    "map50": box.map50,
                    "map": box.map,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "speed": int8_results.speed,
                }
                
                LOGGER.info("\nINT8 Metrics:")
                LOGGER.info(f"  mAP@0.5:      {box.map50:.4f}")
                LOGGER.info(f"  mAP@0.5:0.95: {box.map:.4f}")
                LOGGER.info(f"  Precision:    {prec:.4f}")
                LOGGER.info(f"  Recall:       {rec:.4f}")
                LOGGER.info(f"  F1-Score:     {f1:.4f}")
        except Exception as e:
            LOGGER.error(f"   ✗ INT8 Evaluation failed: {e}")
            traceback.print_exc()
    else:
        LOGGER.info("\n[2/3] Skipping full evaluation, running latency benchmark only...")
        int8_metrics = {}

    # --- LATENCY BENCHMARK ---
    if int8_metrics is not None:
        LOGGER.info("\n  Running manual latency benchmark (CPU)...")
        manual_times = []
        
        # Warmup
        dummy_input = torch.rand(1, 3, imgsz, imgsz).to(int8_device)
        for _ in range(10):
            yolo_int8.predict(dummy_input, verbose=False, device=int8_device, half=False)
            
        # Benchmark
        for _ in range(50):
            t0 = time.perf_counter()
            yolo_int8.predict(dummy_input, verbose=False, device=int8_device, half=False)
            t1 = time.perf_counter()
            manual_times.append((t1 - t0) * 1000)
            
        mean_lat = np.mean(manual_times)
        LOGGER.info(f"  Mean Latency (CPU): {mean_lat:.2f} ms")
        int8_metrics['latency_ms'] = mean_lat

    # --- FP32 COMPARISON ---
    if fp32_checkpoint_path:
        fp32_path = Path(fp32_checkpoint_path)
        if fp32_path.exists():
            LOGGER.info(f"\n[3/3] Evaluating FP32 model: {fp32_path.name}...")
            
            # Clear CUDA cache if using GPU to free memory from any previous ops
            if device != 'cpu':
                torch.cuda.empty_cache()
                
            try:
                yolo_fp32 = YOLO(fp32_checkpoint_path)
                fp32_results = yolo_fp32.val(
                    data=data_cfg,
                    imgsz=imgsz,
                    batch=batch,
                    device=device, # Use requested device (likely GPU)
                    plots=False,
                    save=False,
                    verbose=False
                )
                
                if hasattr(fp32_results, 'box') and int8_metrics:
                    fp32_map = fp32_results.box.map50
                    int8_map = int8_metrics.get('map50', 0)
                    
                    LOGGER.info("\n" + "="*40)
                    LOGGER.info(" COMPARISON RESULTS")
                    LOGGER.info("="*40)
                    LOGGER.info(f"FP32 mAP@0.5: {fp32_map:.4f}")
                    LOGGER.info(f"INT8 mAP@0.5: {int8_map:.4f}")
                    if fp32_map > 0:
                        loss = (fp32_map - int8_map) / fp32_map * 100
                        LOGGER.info(f"Accuracy Loss: {loss:.2f}%")
                        
            except Exception as e:
                LOGGER.warning(f"FP32 evaluation failed: {e}")

    return int8_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--int8', type=str, default='runs/detect/train/weights/best_int8.pt', help='Path to INT8 checkpoint')
    parser.add_argument('--fp32', type=str, default=None, help='Path to FP32 checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default=0, help='Device for FP32 model (INT8 always uses CPU)')
    parser.add_argument('--qat-checkpoint', type=str, help='Source QAT checkpoint to convert')
    parser.add_argument('--export-only', action='store_true', help='Convert QAT to INT8 and exit')
    parser.add_argument('--skip-eval', action='store_true', help='Skip mAP calc, only check speed')
    
    args = parser.parse_args()

    # Handle device arg
    device = args.device
    if str(device).lower() != 'cpu' and str(device).isdigit():
        device = int(device)

    # 1. Convert QAT to INT8 if requested
    target_int8_path = args.int8
    if args.qat_checkpoint:
        out_name = Path(args.qat_checkpoint).stem + '_int8.pt'
        target_int8_path = str(Path(args.qat_checkpoint).parent / out_name)
        try:
            convert_qat_checkpoint(args.qat_checkpoint, target_int8_path, args.imgsz)
            if args.export_only:
                return
        except Exception as e:
            LOGGER.error(f"Conversion failed: {e}")
            traceback.print_exc()
            return

    # 2. Evaluate
    evaluate_int8_model(
        int8_checkpoint_path=target_int8_path,
        fp32_checkpoint_path=args.fp32,
        data_cfg=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        skip_evaluation=args.skip_eval
    )

if __name__ == '__main__':
    main()