#!/usr/bin/env python3
"""
Compare two INT8 models: PTQ vs Hybrid QAT

Usage:
    python compare_int8_models.py <ptq_model.pt> <qat_model.pt> [--data-cfg path/to/data.yaml]
"""

import torch
import sys
from pathlib import Path
from collections import defaultdict

# Add ultralytics path
ultralytics_path = Path(__file__).parent / "obc-yolov8" / "ultralytics10.24"
if ultralytics_path.exists():
    sys.path.insert(0, str(ultralytics_path.parent))
    sys.path.insert(0, str(ultralytics_path))

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Default data config
DEFAULT_DATA_CFG = ultralytics_path / "ultralytics" / "cfg" / "datasets" / "combined_china_motorbike.yaml"


def evaluate_model(checkpoint_path, model_name, data_cfg, imgsz=640, batch=16, device="cpu"):
    """Evaluate an INT8 model and return metrics."""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"Evaluating {model_name}")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get model config from checkpoint or use default
        model_cfg = None
        if 'model_cfg' in checkpoint:
            model_cfg = checkpoint['model_cfg']
        else:
            # Use default model config (yolov8-CA.yaml)
            model_cfg = ultralytics_path / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-CA.yaml"
        
        # For INT8 models, we need to load the model directly from checkpoint
        # YOLO's loading mechanism tries to convert to float, which breaks quantized models
        if 'model' in checkpoint:
            # We have the full model - create YOLO wrapper and assign model directly
            if model_cfg and Path(model_cfg).exists():
                yolo_model = YOLO(str(model_cfg))
            else:
                # Fallback: create minimal YOLO instance
                yolo_model = YOLO('yolov8n.pt')  # Use a standard config as base
            
            # Directly assign the INT8 model (don't let YOLO process it)
            yolo_model.model = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            # Need to reconstruct model first
            if model_cfg and Path(model_cfg).exists():
                yolo_model = YOLO(str(model_cfg))
                try:
                    yolo_model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                except Exception as e:
                    LOGGER.warning(f"Could not load state_dict: {e}")
                    raise
            else:
                raise FileNotFoundError(f"Model config not found: {model_cfg}")
        else:
            raise ValueError("Checkpoint must contain either 'model' or 'model_state_dict'")
        
        # Set to eval mode (handle quantized models carefully)
        # Quantized models may not support .eval() due to structure, but they're already in eval mode
        try:
            yolo_model.model.eval()
            LOGGER.info("Model set to eval mode")
        except (AttributeError, RuntimeError) as e:
            # Quantized models may not support .eval() - that's OK, they're already in eval mode
            LOGGER.info(f"Model already in eval mode or quantized structure (skipping .eval())")
            # Try to set training flag manually if possible
            if hasattr(yolo_model.model, 'training'):
                yolo_model.model.training = False
        
        # Ensure module bookkeeping for quantized models
        # This is critical for quantized Conv2d modules which may be missing hook attributes
        try:
            from ultralytics.nn.tasks import ensure_module_bookkeeping
            from collections import OrderedDict
            import torch.nn as nn
            
            ensure_module_bookkeeping(yolo_model.model, recursive=True)
            
            # Also manually fix hook attributes that quantized modules might be missing
            # PyTorch requires these to be OrderedDict, not regular dict
            hook_attrs = ('_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_backward_pre_hooks',
                         '_state_dict_hooks', '_load_state_dict_pre_hooks')
            
            fixed_count = 0
            for name, module in yolo_model.model.named_modules():
                if isinstance(module, nn.Module):
                    # Initialize missing hook attributes as empty OrderedDicts
                    for hook_attr in hook_attrs:
                        try:
                            if not hasattr(module, hook_attr):
                                object.__setattr__(module, hook_attr, OrderedDict())
                                fixed_count += 1
                            else:
                                attr_value = getattr(module, hook_attr)
                                if not isinstance(attr_value, (dict, OrderedDict)):
                                    object.__setattr__(module, hook_attr, OrderedDict())
                                    fixed_count += 1
                        except (AttributeError, RuntimeError, TypeError):
                            # Some modules may not allow setting these attributes
                            try:
                                object.__setattr__(module, hook_attr, OrderedDict())
                                fixed_count += 1
                            except:
                                pass
                    
                    # Also ensure _non_persistent_buffers_set exists
                    if not hasattr(module, '_non_persistent_buffers_set'):
                        try:
                            object.__setattr__(module, '_non_persistent_buffers_set', set())
                            fixed_count += 1
                        except:
                            pass
                    
                    # Ensure training attribute exists
                    if not hasattr(module, 'training'):
                        try:
                            object.__setattr__(module, 'training', False)
                            fixed_count += 1
                        except:
                            pass
            
            if fixed_count > 0:
                LOGGER.info(f"✓ Fixed {fixed_count} missing/invalid bookkeeping attributes in quantized modules")
            else:
                LOGGER.info("✓ Module bookkeeping verified")
        except Exception as e:
            LOGGER.warning(f"Could not ensure module bookkeeping: {e}")
            import traceback
            LOGGER.debug(traceback.format_exc())
        
        # Determine device and set backend (qnnpack requires CPU)
        backend = checkpoint.get('backend', 'qnnpack')
        
        # CRITICAL: Set quantization backend engine BEFORE evaluation
        # This must match the backend used during conversion
        try:
            from torch.backends import quantized as torch_quantized_backends
            if backend in torch_quantized_backends.supported_engines:
                torch_quantized_backends.engine = backend
                LOGGER.info(f"Set quantization backend engine to {backend}")
            else:
                LOGGER.warning(f"Backend '{backend}' not supported, using default: {torch_quantized_backends.engine}")
        except Exception as e:
            LOGGER.warning(f"Could not set quantization backend: {e}")
        
        if backend == 'qnnpack':
            eval_device = "cpu"
            LOGGER.info(f"Using {eval_device} for evaluation (required for {backend} backend)")
        else:
            eval_device = device
        
        # Run evaluation
        LOGGER.info("Running validation...")
        results = yolo_model.val(
            data=str(data_cfg),
            imgsz=imgsz,
            batch=batch,
            device=eval_device,
            plots=False,
            save=False,
            verbose=True
        )
        
        # Extract metrics
        metrics = {}
        if results:
            if isinstance(results, dict):
                metrics['map50'] = results.get('metrics/mAP50(B)', results.get('map50', None))
                metrics['map'] = results.get('metrics/mAP50-95(B)', results.get('map', None))
                metrics['precision'] = results.get('metrics/precision(B)', results.get('precision', None))
                metrics['recall'] = results.get('metrics/recall(B)', results.get('recall', None))
            else:
                # DetMetrics object
                if hasattr(results, 'box'):
                    metrics['map50'] = getattr(results.box, 'map50', None)
                    metrics['map'] = getattr(results.box, 'map', None)
                    metrics['precision'] = getattr(results.box, 'mp', None)
                    metrics['recall'] = getattr(results.box, 'mr', None)
                else:
                    metrics['map50'] = getattr(results, 'map50', None)
                    metrics['map'] = getattr(results, 'map', None)
                    metrics['precision'] = getattr(results, 'precision', None)
                    metrics['recall'] = getattr(results, 'recall', None)
            
            # Speed metrics
            if hasattr(results, 'speed'):
                metrics['speed'] = results.speed
            elif isinstance(results, dict) and 'speed' in results:
                metrics['speed'] = results['speed']
        
        LOGGER.info(f"\n{model_name} Results:")
        if metrics.get('map50') is not None:
            LOGGER.info(f"  mAP@0.5:      {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)")
        if metrics.get('map') is not None:
            LOGGER.info(f"  mAP@0.5:0.95: {metrics['map']:.4f} ({metrics['map']*100:.2f}%)")
        if metrics.get('precision') is not None:
            LOGGER.info(f"  Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        if metrics.get('recall') is not None:
            LOGGER.info(f"  Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        
        return metrics
        
    except Exception as e:
        LOGGER.error(f"Evaluation failed for {model_name}: {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return None


def compare_models(ptq_path, qat_path, data_cfg=None, imgsz=640, batch=16, device="cpu"):
    """Compare PTQ and QAT INT8 models."""
    
    if data_cfg is None:
        data_cfg = DEFAULT_DATA_CFG
    
    if not Path(data_cfg).exists():
        raise FileNotFoundError(f"Data config not found: {data_cfg}")
    
    LOGGER.info("=" * 80)
    LOGGER.info("Comparing INT8 Models: PTQ vs Hybrid QAT")
    LOGGER.info("=" * 80)
    
    # Evaluate PTQ model
    ptq_metrics = evaluate_model(
        ptq_path,
        "PTQ Model (Post-Training Quantization)",
        data_cfg,
        imgsz=imgsz,
        batch=batch,
        device=device
    )
    
    # Evaluate QAT model
    qat_metrics = evaluate_model(
        qat_path,
        "Hybrid QAT Model (QAT on sensitive modules)",
        data_cfg,
        imgsz=imgsz,
        batch=batch,
        device=device
    )
    
    # Compare results
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Comparison Summary")
    LOGGER.info("=" * 80)
    
    if ptq_metrics and qat_metrics:
        print(f"\n{'Metric':<20} {'PTQ':<15} {'QAT':<15} {'Improvement':<15}")
        print("-" * 65)
        
        for metric_name in ['map50', 'map', 'precision', 'recall']:
            ptq_val = ptq_metrics.get(metric_name)
            qat_val = qat_metrics.get(metric_name)
            
            if ptq_val is not None and qat_val is not None:
                improvement = (qat_val - ptq_val) * 100
                improvement_str = f"{improvement:+.2f}%"
                print(f"{metric_name:<20} {ptq_val:<15.4f} {qat_val:<15.4f} {improvement_str:<15}")
            elif ptq_val is not None:
                print(f"{metric_name:<20} {ptq_val:<15.4f} {'N/A':<15} {'N/A':<15}")
            elif qat_val is not None:
                print(f"{metric_name:<20} {'N/A':<15} {qat_val:<15.4f} {'N/A':<15}")
        
        # Overall improvement
        if ptq_metrics.get('map') and qat_metrics.get('map'):
            overall_improvement = (qat_metrics['map'] - ptq_metrics['map']) * 100
            LOGGER.info(f"\n📊 Overall mAP@0.5:0.95 Improvement: {overall_improvement:+.2f}%")
            
            if overall_improvement > 0:
                LOGGER.info(f"   ✓ QAT model is {overall_improvement:.2f}% better than PTQ")
            elif overall_improvement < 0:
                LOGGER.info(f"   ⚠️  QAT model is {abs(overall_improvement):.2f}% worse than PTQ")
            else:
                LOGGER.info(f"   = Models perform similarly")
    
    return {
        'ptq': ptq_metrics,
        'qat': qat_metrics
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_int8_models.py <ptq_model.pt> <qat_model.pt> [--data-cfg path/to/data.yaml]")
        sys.exit(1)
    
    ptq_path = sys.argv[1]
    qat_path = sys.argv[2]
    data_cfg = None
    
    if '--data-cfg' in sys.argv:
        idx = sys.argv.index('--data-cfg')
        if idx + 1 < len(sys.argv):
            data_cfg = sys.argv[idx + 1]
    
    try:
        compare_models(ptq_path, qat_path, data_cfg)
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

