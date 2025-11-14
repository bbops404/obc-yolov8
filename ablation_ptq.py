"""
Ablation study for PTQ quantization of different model components.
Tests various combinations of quantizing: backbone, neck, BoTNet, CoordAtt.
"""

import sys
from pathlib import Path

# Add local ultralytics to path BEFORE importing ultralytics
sys.path.insert(0, str(Path(__file__).parent / 'obc-yolov8' / 'ultralytics10.24'))

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from train_ptq import train_ptq, DEFAULT_MODEL_CFG, DEFAULT_DATA_CFG
import json
from datetime import datetime


def run_ablation_study(
    weights: str | Path,
    model_cfg: str | Path = DEFAULT_MODEL_CFG,
    data_cfg: str | Path = DEFAULT_DATA_CFG,
    imgsz: int = 640,
    batch: int = 1,
    workers: int = 4,
    device: str = "cpu",
    backend: str = "fbgemm",
    num_calibration_batches: int = 5,
    calibration_split: str = "val",
    save_dir: Path | None = None,
):
    """
    Run ablation study testing different quantization combinations.
    
    Args:
        weights: Path to FP32 model weights
        model_cfg: Model configuration YAML
        data_cfg: Dataset configuration YAML
        imgsz: Image size
        batch: Batch size for evaluation
        workers: Number of data loader workers
        device: Device to use (should be 'cpu' for fbgemm backend)
        backend: Quantization backend
        num_calibration_batches: Number of batches for calibration
        calibration_split: Dataset split for calibration ('val' or 'train')
        save_dir: Directory to save results
    """
    
    # Define ablation configurations
    # Each config specifies which components to quantize
    ablation_configs = [
        {
            "name": "baseline_fp32",
            "description": "FP32 baseline (no quantization)",
            "quantize_backbone": False,
            "quantize_neck": False,
            "quantize_botnet": False,
            "quantize_coordatt": False,
        },
        {
            "name": "backbone_only",
            "description": "Backbone only",
            "quantize_backbone": True,
            "quantize_neck": False,
            "quantize_botnet": False,
            "quantize_coordatt": False,
        },
        {
            "name": "neck_only",
            "description": "Neck only",
            "quantize_backbone": False,
            "quantize_neck": True,
            "quantize_botnet": False,
            "quantize_coordatt": False,
        },
        {
            "name": "botnet_only",
            "description": "BoTNet only",
            "quantize_backbone": False,
            "quantize_neck": False,
            "quantize_botnet": True,
            "quantize_coordatt": False,
        },
        {
            "name": "coordatt_only",
            "description": "CoordAtt only",
            "quantize_backbone": False,
            "quantize_neck": False,
            "quantize_botnet": False,
            "quantize_coordatt": True,
        },
        {
            "name": "backbone_neck",
            "description": "Backbone + Neck",
            "quantize_backbone": True,
            "quantize_neck": True,
            "quantize_botnet": False,
            "quantize_coordatt": False,
        },
        {
            "name": "backbone_botnet",
            "description": "Backbone + BoTNet",
            "quantize_backbone": True,
            "quantize_neck": False,
            "quantize_botnet": True,
            "quantize_coordatt": False,
        },
        {
            "name": "backbone_coordatt",
            "description": "Backbone + CoordAtt",
            "quantize_backbone": True,
            "quantize_neck": False,
            "quantize_botnet": False,
            "quantize_coordatt": True,
        },
        {
            "name": "neck_botnet",
            "description": "Neck + BoTNet",
            "quantize_backbone": False,
            "quantize_neck": True,
            "quantize_botnet": True,
            "quantize_coordatt": False,
        },
        {
            "name": "neck_coordatt",
            "description": "Neck + CoordAtt",
            "quantize_backbone": False,
            "quantize_neck": True,
            "quantize_botnet": False,
            "quantize_coordatt": True,
        },
        {
            "name": "botnet_coordatt",
            "description": "BoTNet + CoordAtt",
            "quantize_backbone": False,
            "quantize_neck": False,
            "quantize_botnet": True,
            "quantize_coordatt": True,
        },
        {
            "name": "backbone_neck_botnet",
            "description": "Backbone + Neck + BoTNet",
            "quantize_backbone": True,
            "quantize_neck": True,
            "quantize_botnet": True,
            "quantize_coordatt": False,
        },
        {
            "name": "backbone_neck_coordatt",
            "description": "Backbone + Neck + CoordAtt",
            "quantize_backbone": True,
            "quantize_neck": True,
            "quantize_botnet": False,
            "quantize_coordatt": True,
        },
        {
            "name": "backbone_botnet_coordatt",
            "description": "Backbone + BoTNet + CoordAtt",
            "quantize_backbone": True,
            "quantize_neck": False,
            "quantize_botnet": True,
            "quantize_coordatt": True,
        },
        {
            "name": "neck_botnet_coordatt",
            "description": "Neck + BoTNet + CoordAtt",
            "quantize_backbone": False,
            "quantize_neck": True,
            "quantize_botnet": True,
            "quantize_coordatt": True,
        },
        {
            "name": "all_components",
            "description": "All components (Backbone + Neck + BoTNet + CoordAtt)",
            "quantize_backbone": True,
            "quantize_neck": True,
            "quantize_botnet": True,
            "quantize_coordatt": True,
        },
    ]
    
    results = []
    save_dir = Path(save_dir) if save_dir else Path("runs/detect/ptq_ablation")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("=" * 80)
    LOGGER.info("PTQ Ablation Study")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Total configurations to test: {len(ablation_configs)}")
    LOGGER.info(f"Results will be saved to: {save_dir}")
    LOGGER.info("")
    
    for idx, config in enumerate(ablation_configs, 1):
        LOGGER.info("=" * 80)
        LOGGER.info(f"Configuration {idx}/{len(ablation_configs)}: {config['name']}")
        LOGGER.info(f"Description: {config['description']}")
        LOGGER.info("=" * 80)
        
        try:
            # For baseline (FP32), skip quantization
            if config['name'] == 'baseline_fp32':
                LOGGER.info("Skipping quantization for FP32 baseline")
                # Load and evaluate FP32 model directly
                model = YOLO(str(weights))
                eval_results = model.val(
                    data=str(data_cfg),
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    plots=False,
                    save=False,
                    verbose=False,
                )
                
                # Extract metrics
                if isinstance(eval_results, dict):
                    map50 = eval_results.get('metrics/mAP50(B)', eval_results.get('map50', None))
                    map = eval_results.get('metrics/mAP50-95(B)', eval_results.get('map', None))
                    precision = eval_results.get('metrics/precision(B)', eval_results.get('precision', None))
                    recall = eval_results.get('metrics/recall(B)', eval_results.get('recall', None))
                else:
                    map50 = getattr(eval_results, 'map50', None)
                    map = getattr(eval_results, 'map', None)
                    precision = getattr(eval_results, 'precision', None)
                    recall = getattr(eval_results, 'recall', None)
                
                result = {
                    "config_name": config['name'],
                    "description": config['description'],
                    "quantize_backbone": False,
                    "quantize_neck": False,
                    "quantize_botnet": False,
                    "quantize_coordatt": False,
                    "map50": float(map50) if map50 is not None else None,
                    "map": float(map) if map is not None else None,
                    "precision": float(precision) if precision is not None else None,
                    "recall": float(recall) if recall is not None else None,
                    "int8_path": None,
                    "status": "success",
                }
            else:
                # Run PTQ with specific configuration
                # We need to modify prepare_for_ptq to accept quantization config
                # For now, we'll use a workaround by modifying the method temporarily
                
                # Import and patch prepare_for_ptq to accept config
                from ultralytics.nn.tasks import DetectionModel
                original_prepare = DetectionModel.prepare_for_ptq
                
                def patched_prepare_for_ptq(self, backend='fbgemm', example_input=None, use_fx=True,
                                           quantize_backbone=True, quantize_neck=True,
                                           quantize_botnet=True, quantize_coordatt=True):
                    """Patched version that accepts quantization config."""
                    return original_prepare(self, backend, example_input, use_fx,
                                          quantize_backbone=quantize_backbone,
                                          quantize_neck=quantize_neck,
                                          quantize_botnet=quantize_botnet,
                                          quantize_coordatt=quantize_coordatt)
                
                # Temporarily replace the method
                DetectionModel.prepare_for_ptq = patched_prepare_for_ptq
                
                # Run PTQ
                run_name = f"ablation_{config['name']}"
                ptq_result = train_ptq(
                    weights=weights,
                    model_cfg=model_cfg,
                    data_cfg=data_cfg,
                    imgsz=imgsz,
                    batch=batch,
                    workers=workers,
                    device=device,
                    backend=backend,
                    save_dir=save_dir,
                    run_name=run_name,
                    convert_to_int8=True,
                    use_fx=True,
                    num_calibration_batches=num_calibration_batches,
                    calibration_split=calibration_split,
                    evaluate=True,
                    quantize_backbone=config['quantize_backbone'],
                    quantize_neck=config['quantize_neck'],
                    quantize_botnet=config['quantize_botnet'],
                    quantize_coordatt=config['quantize_coordatt'],
                )
                
                # Restore original method
                DetectionModel.prepare_for_ptq = original_prepare
                
                # Extract metrics from evaluation
                int8_path = ptq_result.get('int8_path')
                
                # Load and evaluate the INT8 model to get metrics
                if int8_path and Path(int8_path).exists():
                    int8_model = YOLO(str(int8_path))
                    eval_results = int8_model.val(
                        data=str(data_cfg),
                        imgsz=imgsz,
                        batch=batch,
                        device=device,
                        plots=False,
                        save=False,
                        verbose=False,
                    )
                    
                    # Extract metrics
                    if isinstance(eval_results, dict):
                        map50 = eval_results.get('metrics/mAP50(B)', eval_results.get('map50', None))
                        map = eval_results.get('metrics/mAP50-95(B)', eval_results.get('map', None))
                        precision = eval_results.get('metrics/precision(B)', eval_results.get('precision', None))
                        recall = eval_results.get('metrics/recall(B)', eval_results.get('recall', None))
                    else:
                        map50 = getattr(eval_results, 'map50', None)
                        map = getattr(eval_results, 'map', None)
                        precision = getattr(eval_results, 'precision', None)
                        recall = getattr(eval_results, 'recall', None)
                else:
                    map50 = map = precision = recall = None
                
                result = {
                    "config_name": config['name'],
                    "description": config['description'],
                    "quantize_backbone": config['quantize_backbone'],
                    "quantize_neck": config['quantize_neck'],
                    "quantize_botnet": config['quantize_botnet'],
                    "quantize_coordatt": config['quantize_coordatt'],
                    "map50": float(map50) if map50 is not None else None,
                    "map": float(map) if map is not None else None,
                    "precision": float(precision) if precision is not None else None,
                    "recall": float(recall) if recall is not None else None,
                    "int8_path": str(int8_path) if int8_path else None,
                    "status": "success",
                }
            
            results.append(result)
            
            LOGGER.info(f"✓ Configuration {config['name']} completed")
            if result['map50'] is not None:
                LOGGER.info(f"  mAP@0.5: {result['map50']:.4f} ({result['map50']*100:.2f}%)")
            if result['map'] is not None:
                LOGGER.info(f"  mAP@0.5:0.95: {result['map']:.4f} ({result['map']*100:.2f}%)")
            if result['precision'] is not None:
                LOGGER.info(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)")
            if result['recall'] is not None:
                LOGGER.info(f"  Recall: {result['recall']:.4f} ({result['recall']*100:.2f}%)")
            LOGGER.info("")
            
        except Exception as e:
            LOGGER.error(f"✗ Configuration {config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "config_name": config['name'],
                "description": config['description'],
                "quantize_backbone": config.get('quantize_backbone', False),
                "quantize_neck": config.get('quantize_neck', False),
                "quantize_botnet": config.get('quantize_botnet', False),
                "quantize_coordatt": config.get('quantize_coordatt', False),
                "map50": None,
                "map": None,
                "precision": None,
                "recall": None,
                "int8_path": None,
                "status": "failed",
                "error": str(e),
            })
            LOGGER.info("")
    
    # Save results to JSON
    results_file = save_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "weights": str(weights),
            "model_cfg": str(model_cfg),
            "data_cfg": str(data_cfg),
            "imgsz": imgsz,
            "backend": backend,
            "num_calibration_batches": num_calibration_batches,
            "calibration_split": calibration_split,
            "results": results,
        }, f, indent=2)
    
    LOGGER.info("=" * 80)
    LOGGER.info("Ablation Study Complete")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Results saved to: {results_file}")
    LOGGER.info("")
    LOGGER.info("Summary:")
    LOGGER.info("-" * 80)
    
    # Print summary table
    baseline_result = next((r for r in results if r['config_name'] == 'baseline_fp32'), None)
    baseline_map50 = baseline_result['map50'] if baseline_result else None
    
    LOGGER.info(f"{'Configuration':<30} {'mAP@0.5':>10} {'mAP@0.5:0.95':>12} {'Precision':>10} {'Recall':>10} {'ΔmAP@0.5':>10}")
    LOGGER.info("-" * 80)
    
    for result in results:
        config_name = result['config_name']
        map50 = result['map50']
        map_val = result['map']
        precision = result['precision']
        recall = result['recall']
        
        if baseline_map50 is not None and map50 is not None:
            delta = map50 - baseline_map50
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"
        
        map50_str = f"{map50:.4f}" if map50 is not None else "N/A"
        map_str = f"{map_val:.4f}" if map_val is not None else "N/A"
        precision_str = f"{precision:.4f}" if precision is not None else "N/A"
        recall_str = f"{recall:.4f}" if recall is not None else "N/A"
        
        LOGGER.info(f"{config_name:<30} {map50_str:>10} {map_str:>12} {precision_str:>10} {recall_str:>10} {delta_str:>10}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PTQ Ablation Study")
    parser.add_argument("--weights", type=str, required=True, help="Path to FP32 model weights")
    parser.add_argument("--model-cfg", type=str, default=str(DEFAULT_MODEL_CFG), help="Model configuration YAML")
    parser.add_argument("--data-cfg", type=str, default=str(DEFAULT_DATA_CFG), help="Dataset configuration YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--backend", type=str, default="fbgemm", help="Quantization backend")
    parser.add_argument("--num-calibration-batches", type=int, default=None, help="Number of batches for calibration (None = use entire validation set)")
    parser.add_argument("--calibration-split", type=str, default="val", choices=["val", "train"], help="Dataset split for calibration")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save results")
    
    args = parser.parse_args()
    
    run_ablation_study(
        weights=args.weights,
        model_cfg=args.model_cfg,
        data_cfg=args.data_cfg,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        backend=args.backend,
        num_calibration_batches=args.num_calibration_batches,
        calibration_split=args.calibration_split,
        save_dir=args.save_dir,
    )

