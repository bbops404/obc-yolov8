#!/usr/bin/env python3
"""Compare FP32 and INT8 model metrics."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

def evaluate_model(checkpoint_path: str, data_cfg: str, model_name: str):
    """Evaluate a model and return metrics."""
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Evaluating {model_name}: {checkpoint_path}")
    LOGGER.info(f"{'='*80}")
    
    try:
        model = YOLO(checkpoint_path)
        
        # ------------------- FIX START -------------------
        if model.model is None:
            raise AttributeError("YOLO model failed to initialize its internal structure (model.model is None). This usually happens when loading a non-standard or manually quantized checkpoint. Please ensure your custom script's loading logic is incorporated if needed.")
        # ------------------- FIX END -------------------
        
        model.model.eval() # Now safe to call
        
        # For INT8 models, ensure we're on CPU if using fbgemm
        if 'int8' in checkpoint_path.lower() or 'quantized' in checkpoint_path.lower():
            device = 'cpu'
        else:
            device = 0
        
        results = model.val(data=data_cfg, imgsz=640, batch=16, device=device, half=False, plots=False, save=False, verbose=True)
        
        if hasattr(results, 'box'):
            metrics = {
                'map50': results.box.map50,
                'map': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'speed': getattr(results, 'speed', {})
            }
            return metrics
        return None
    except Exception as e:
        LOGGER.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    data_cfg = str(ULTRALYTICS_PATH / "ultralytics" / "cfg" / "datasets" / "combined_china_motorbike.yaml")
    
    fp32_path = "/Users/user/Documents/obc-yolov8/obc-yolov8/runs/detect/train7/weights/last.pt"
    int8_path = "/Users/user/Documents/obc-yolov8/runs/detect/train_ptq/weights/best_int8.pt"

    LOGGER.info("Comparing FP32 vs INT8 Model Performance")
    LOGGER.info("="*80)
    
    # Evaluate FP32
    fp32_metrics = evaluate_model(fp32_path, data_cfg, "FP32 Model")
    
    # Evaluate INT8
    int8_metrics = evaluate_model(int8_path, data_cfg, "INT8 Model")
    
    # Compare
    if fp32_metrics and int8_metrics:
        LOGGER.info("\n" + "="*80)
        LOGGER.info("COMPARISON: FP32 vs INT8")
        LOGGER.info("="*80)
        
        metrics_to_compare = [
            ('mAP@0.5', 'map50'),
            ('mAP@0.5:0.95', 'map'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
        ]
        
        LOGGER.info(f"{'Metric':<20} {'FP32':<15} {'INT8':<15} {'Difference':<15} {'Change':<10}")
        LOGGER.info("-"*80)
        
        for display_name, key in metrics_to_compare:
            fp32_val = fp32_metrics[key]
            int8_val = int8_metrics[key]
            diff = int8_val - fp32_val
            change_pct = (diff / fp32_val * 100) if fp32_val > 0 else 0
            change_str = f"{change_pct:+.2f}%"
            
            LOGGER.info(f"{display_name:<20} {fp32_val:<15.4f} {int8_val:<15.4f} {diff:+.4f} ({change_str:<10})")
        
        # Speed comparison
        if 'speed' in fp32_metrics and 'speed' in int8_metrics:
            fp32_inf = fp32_metrics['speed'].get('inference', 0)
            int8_inf = int8_metrics['speed'].get('inference', 0)
            if fp32_inf > 0 and int8_inf > 0:
                speedup = fp32_inf / int8_inf
                LOGGER.info(f"\n{'Inference Speed':<20} {fp32_inf:<15.2f}ms {int8_inf:<15.2f}ms {speedup:.2f}x {'(CPU)' if int8_inf > fp32_inf else '(GPU)'}")
        
        # Overall accuracy retention
        if fp32_metrics['map50'] > 0:
            retention = (int8_metrics['map50'] / fp32_metrics['map50']) * 100
            LOGGER.info(f"\n{'Accuracy Retention (mAP@0.5)':<20} {retention:.2f}%")
        
        LOGGER.info("\n" + "="*80)
    else:
        LOGGER.error("Failed to get metrics from one or both models")

