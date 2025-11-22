"""Hybrid Quantization: PTQ + Targeted QAT (Starting from FP32).

This script loads an FP32 model, applies PTQ preparation (calibration),
then performs QAT fine-tuning on sensitive modules only.

This version works with your converted INT8 model by starting fresh from FP32.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, cast

import torch
import torch.nn as nn
from torch.backends import quantized as torch_quantized_backends
from tqdm import tqdm


# Ensure the local ultralytics package is importable
REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if not ULTRALYTICS_PATH.exists():
    ULTRALYTICS_PATH = REPO_ROOT / "ultralytics10.24"
import sys

if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO, __version__
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel

ensure_module_bookkeeping = getattr(
    ultralytics_tasks,
    "ensure_module_bookkeeping",
    lambda *args, **kwargs: None,
)


DEFAULT_PROJECT = REPO_ROOT / "runs" / "detect"
DEFAULT_NAME = "train_hybrid_qat"
DEFAULT_MODEL_CFG = ULTRALYTICS_PATH / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-CA.yaml"
DEFAULT_DATA_CFG = ULTRALYTICS_PATH / "ultralytics" / "cfg" / "datasets" / "combined_china_motorbike.yaml"


def _resolve_device(device: Any) -> str:
    """Convert device to string format."""
    if isinstance(device, torch.device):
        return str(device)
    if isinstance(device, (list, tuple)):
        return ",".join(map(str, device))
    return str(device)


def freeze_all_parameters(model):
    """Freeze all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_sensitive_modules(model, sensitive_modules: List[str], logger=None):
    """Unfreeze only the specified sensitive modules."""
    if logger is None:
        logger = LOGGER
    
    trainable_params = []
    unfrozen_modules = []
    
    for name, module in model.named_modules():
        is_sensitive = any(
            pattern.lower() in name.lower() 
            for pattern in sensitive_modules
        )
        
        if is_sensitive:
            logger.info(f"Unfreezing sensitive module: {name}")
            unfrozen_modules.append(name)
            
            for param in module.parameters():
                param.requires_grad = True
                if param not in trainable_params:
                    trainable_params.append(param)
    
    if not unfrozen_modules:
        logger.warning(f"⚠️  No modules matched patterns: {sensitive_modules}")
    
    return trainable_params


def train_hybrid_qat(
    fp32_weights: str | Path,
    model_cfg: str | Path = DEFAULT_MODEL_CFG,
    data_cfg: str | Path = DEFAULT_DATA_CFG,
    imgsz: int = 640,
    batch: Optional[int] = None,
    workers: Optional[int] = None,
    device: Any = 0,
    backend: str = "qnnpack",
    save_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    epochs: int = 5,
    lr: float = 1e-4,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    sensitive_modules: Optional[List[str]] = None,
    num_calibration_batches: Optional[int] = None,
    quantize_botnet: bool = True,
    quantize_coordatt: bool = True,
    evaluate_before: bool = True,
    evaluate_during: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Hybrid QAT starting from FP32 model.

    This version:
    1. Loads FP32 model
    2. Applies PTQ preparation (calibration)
    3. Fine-tunes with QAT on sensitive modules

    Args:
        fp32_weights: Path to trained FP32 model weights
        model_cfg: Path to model YAML configuration
        data_cfg: Path to dataset YAML configuration
        imgsz: Image size for training
        batch: Batch size
        workers: Number of data loading workers
        device: Device to use
        backend: Quantization backend
        save_dir: Directory to save outputs
        run_name: Name for the run
        epochs: Number of fine-tuning epochs
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: Weight decay
        sensitive_modules: Module patterns to fine-tune
        num_calibration_batches: Batches for PTQ calibration
        quantize_botnet: Whether to quantize BoTNet (if False, stays FP32)
        quantize_coordatt: Whether to quantize CoordAtt (if False, stays FP32)
        evaluate_before: Evaluate after PTQ, before QAT
        evaluate_during: Evaluate after each QAT epoch

    Returns:
        Dictionary with paths and metrics
    """

    project_dir = Path(save_dir) if save_dir is not None else DEFAULT_PROJECT
    run_name = run_name or DEFAULT_NAME
    project_dir.mkdir(parents=True, exist_ok=True)
    
    if sensitive_modules is None:
        sensitive_modules = ['model.10', 'model.19', 'model.20', 'model.23', 'model.24']
    
    LOGGER.info("=" * 80)
    LOGGER.info("HYBRID QAT: PTQ Preparation + Targeted Fine-tuning")
    LOGGER.info("=" * 80)
    LOGGER.info(f"FP32 weights:     {fp32_weights}")
    LOGGER.info(f"Sensitive modules: {', '.join(sensitive_modules)}")
    LOGGER.info(f"Quantize BoTNet:   {quantize_botnet}")
    LOGGER.info(f"Quantize CoordAtt: {quantize_coordatt}")
    LOGGER.info(f"Epochs:           {epochs}")
    LOGGER.info(f"Learning rate:    {lr}")
    LOGGER.info("=" * 80)

    # Set backend early
    if backend in torch_quantized_backends.supported_engines:
        torch_quantized_backends.engine = backend
        LOGGER.info(f"Set quantization backend to {backend}")

    # Load FP32 model robustly: instantiate from config then load weights
    LOGGER.info(f"\nLoading FP32 model from {fp32_weights}...")
    base_model = YOLO(str(model_cfg))

    fp_path = Path(fp32_weights)
    ckpt = None
    if fp_path.exists():
        try:
            ckpt = torch.load(fp_path, map_location='cpu')
        except Exception:
            ckpt = None

    # If checkpoint-like object found, try to apply it to the reconstructed model
    if isinstance(ckpt, dict):
        loaded = ckpt.get('model', None)
        if loaded is None and 'model_state_dict' in ckpt:
            loaded = ckpt['model_state_dict']

        if isinstance(loaded, nn.Module):
            base_model.model = loaded
        elif isinstance(loaded, dict):
            inner = getattr(base_model, 'model', None)
            if not isinstance(inner, nn.Module):
                raise RuntimeError('YOLO did not create an inner model to load state_dict into')
            inner.load_state_dict(loaded)
        elif isinstance(loaded, (str, Path)):
            base_model = YOLO(model=str(loaded))
        else:
            # If checkpoint appears to be a bare state_dict (keys->tensors), load it
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                inner = getattr(base_model, 'model', None)
                if not isinstance(inner, nn.Module):
                    raise RuntimeError('YOLO did not create an inner model to load state_dict into')
                inner.load_state_dict(ckpt)
            # else: fallback to trying YOLO(fp32_weights) below

    # If no usable checkpoint was loaded, try to construct YOLO directly from fp32_weights
    model_candidate = getattr(base_model, 'model', None)
    if not isinstance(model_candidate, nn.Module):
        try:
            alt = YOLO(str(fp32_weights))
            if isinstance(getattr(alt, 'model', None), nn.Module):
                base_model = alt
        except Exception:
            pass

    model = base_model
    detection_model = getattr(model, 'model', None)

    if not isinstance(detection_model, nn.Module) or not hasattr(detection_model, 'prepare_for_ptq'):
        raise AttributeError("Model doesn't have prepare_for_ptq method or inner model couldn't be loaded")

    LOGGER.info("✓ FP32 model loaded")

    # Prepare device
    device_str = _resolve_device(device)
    if backend in ['qnnpack']:
        torch_device = torch.device("cpu")
        device_str = "cpu"
        LOGGER.info(f"Using CPU (required for {backend})")
    else:
        if device_str == "cpu":
            torch_device = torch.device("cpu")
        elif device_str.isdigit() and torch.cuda.is_available():
            torch_device = torch.device(f"cuda:{device_str}")
        else:
            torch_device = torch.device("cpu")
            device_str = "cpu"
    
    model.to(torch_device)

    # Load calibration data
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Step 1: PTQ Preparation (Calibration)")
    LOGGER.info("=" * 80)
    
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.cfg import get_cfg
    
    dataset = check_det_dataset(str(data_cfg))
    cfg_args = get_cfg(overrides={
        "imgsz": imgsz,
        "batch": batch or 16,
        "workers": workers or 8,
        "device": device_str,
        "task": "detect",
    })
    
    # Build calibration dataset (using validation set)
    cal_dataset = build_yolo_dataset(
        cfg=cfg_args,
        img_path=dataset.get("val", ""),
        batch=batch or 16,
        data=dataset,
        mode="val",
        rect=False,
        stride=32,
    )
    cal_loader = build_dataloader(
        dataset=cal_dataset,
        batch=batch or 16,
        workers=workers or 8,
        shuffle=False,
        rank=-1,
    )
    LOGGER.info(f"Calibration dataset: {len(cal_loader)} batches")

    # Prepare model for PTQ
    LOGGER.info("Preparing model for quantization...")
    example_input = torch.randn(1, 3, imgsz, imgsz)
    
    detection_model_any = cast(Any, detection_model)
    prepared_model = detection_model_any.prepare_for_ptq(
        backend=backend,
        example_input=example_input,
        use_fx=False,  # Use eager mode for better compatibility
        quantize_backbone=True,
        quantize_neck=True,
        quantize_botnet=quantize_botnet,
        quantize_coordatt=quantize_coordatt,
    )
    
    # Manual qconfig clearing for modules to keep in FP32
    if not quantize_botnet:
        LOGGER.info("Keeping BoTNet (model.10) in FP32...")
        try:
            botnet = prepared_model.model.get_submodule('10')
            botnet.qconfig = None
            for m in botnet.modules():
                if hasattr(m, 'qconfig'):
                    m.qconfig = None
        except:
            LOGGER.warning("Could not clear BoTNet qconfig")
    
    if not quantize_coordatt:
        LOGGER.info("Keeping CoordAtt in FP32...")
        for idx in ['19', '20', '23', '24']:
            try:
                ca = prepared_model.model.get_submodule(idx)
                ca.qconfig = None
                for m in ca.modules():
                    if hasattr(m, 'qconfig'):
                        m.qconfig = None
            except:
                pass
    
    model.model = prepared_model

    # Calibrate
    LOGGER.info("Calibrating with validation data...")
    calibrated_model = prepared_model.calibrate_ptq(
        calibration_data=cal_loader,
        num_batches=num_calibration_batches,
    )
    model.model = calibrated_model
    LOGGER.info("✓ PTQ calibration complete")

    # Evaluate PTQ baseline
    ptq_map = None
    if evaluate_before:
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("Evaluating PTQ baseline...")
        LOGGER.info("=" * 80)
        try:
            eval_results = model.val(
                data=str(data_cfg),
                imgsz=imgsz,
                batch=batch or 16,
                device=device_str,
                plots=False,
                save=False,
                verbose=True
            )
            
            if eval_results:
                ptq_map = getattr(eval_results, 'map', None)
                if ptq_map is None and hasattr(eval_results, 'results_dict'):
                    ptq_map = eval_results.results_dict.get('metrics/mAP50-95(B)', None)
                
                map50 = getattr(eval_results, 'map50', None)
                
                LOGGER.info("\nPTQ Baseline:")
                if map50:
                    LOGGER.info(f"  mAP@0.5:      {map50:.4f} ({map50*100:.2f}%)")
                if ptq_map:
                    LOGGER.info(f"  mAP@0.5:0.95: {ptq_map:.4f} ({ptq_map*100:.2f}%)")
        except Exception as e:
            LOGGER.warning(f"PTQ evaluation failed: {e}")

    # Prepare for QAT
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Step 2: QAT Fine-tuning on Sensitive Modules")
    LOGGER.info("=" * 80)
    
    calibrated_model.train()
    calibrated_model.apply(torch.quantization.disable_observer)
    
    freeze_all_parameters(calibrated_model)
    trainable_params = unfreeze_sensitive_modules(calibrated_model, sensitive_modules, LOGGER)
    
    if not trainable_params:
        raise ValueError("No trainable parameters!")
    
    total_params = sum(p.numel() for p in calibrated_model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    LOGGER.info(f"Trainable: {trainable_count:,} / {total_params:,} ({100*trainable_count/total_params:.2f}%)")

    # Setup optimizer
    optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Load training data
    train_dataset = build_yolo_dataset(
        cfg=cfg_args,
        img_path=dataset.get("train", ""),
        batch=batch or 16,
        data=dataset,
        mode="train",
        rect=False,
        stride=32,
    )
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch=batch or 16,
        workers=workers or 8,
        shuffle=True,
        rank=-1,
    )
    LOGGER.info(f"Training dataset: {len(train_loader)} batches")

    # Setup save directory
    save_dir_path = project_dir / run_name
    weights_dir = save_dir_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"Starting QAT fine-tuning for {epochs} epochs...")
    LOGGER.info("=" * 80)
    
    best_map = ptq_map if ptq_map else 0.0
    best_epoch = -1

    for epoch in range(epochs):
        LOGGER.info(f"\nEpoch {epoch+1}/{epochs}")
        calibrated_model.train()
        
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch_data in enumerate(pbar):
            try:
                images = batch_data['img'].to(torch_device).float() / 255.0
                
                # Forward
                preds = calibrated_model(images)
                
                # Compute loss using model's built-in method
                if hasattr(calibrated_model, 'compute_loss'):
                    loss, loss_items = calibrated_model.compute_loss(batch_data, preds)
                elif hasattr(calibrated_model, 'loss'):
                    loss, loss_items = calibrated_model.loss(batch_data, preds)
                else:
                    raise RuntimeError("Model has no loss computation method!")
                
                if not torch.isfinite(loss):
                    continue
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                LOGGER.warning(f"Batch {batch_idx} failed: {e}")
                continue
        
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        LOGGER.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        scheduler.step()
        
        # Evaluate
        if evaluate_during:
            try:
                eval_results = model.val(
                    data=str(data_cfg),
                    imgsz=imgsz,
                    batch=batch or 16,
                    device=device_str,
                    plots=False,
                    save=False,
                    verbose=False
                )
                
                if eval_results:
                    current_map = getattr(eval_results, 'map', None)
                    if current_map is None and hasattr(eval_results, 'results_dict'):
                        current_map = eval_results.results_dict.get('metrics/mAP50-95(B)', None)
                    
                    if current_map:
                        LOGGER.info(f"  mAP: {current_map:.4f} ({current_map*100:.2f}%)")
                        
                        if current_map > best_map:
                            best_map = current_map
                            best_epoch = epoch + 1
                            
                            torch.save({
                                "model": calibrated_model,
                                "epoch": epoch + 1,
                                "best_fitness": current_map,
                                "hybrid_qat": True,
                            }, weights_dir / "best.pt")
                            LOGGER.info(f"  ✓ New best!")
            except Exception as e:
                LOGGER.warning(f"Evaluation failed: {e}")

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("QAT Complete!")
    LOGGER.info("=" * 80)
    if ptq_map and best_map > 0:
        improvement = (best_map - ptq_map) * 100
        LOGGER.info(f"PTQ:        {ptq_map:.4f}")
        LOGGER.info(f"Best QAT:   {best_map:.4f}")
        LOGGER.info(f"Improvement: +{improvement:.2f}%")
    LOGGER.info("=" * 80)

    return {
        "best_path": weights_dir / "best.pt",
        "best_map": best_map,
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid QAT from FP32")
    parser.add_argument("--fp32-weights", type=str, required=True, help="Path to FP32 model")
    parser.add_argument("--model-cfg", type=str, default=str(DEFAULT_MODEL_CFG))
    parser.add_argument("--data-cfg", type=str, default=str(DEFAULT_DATA_CFG))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--backend", type=str, default="qnnpack", choices=["qnnpack", "fbgemm"])
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-calibration-batches", type=int, default=None)
    parser.add_argument("--skip-botnet-quant", action="store_true")
    parser.add_argument("--skip-ca-quant", action="store_true")
    parser.add_argument("--sensitive-modules", nargs='+', default=None)

    args = parser.parse_args()

    results = train_hybrid_qat(
        fp32_weights=args.fp32_weights,
        model_cfg=args.model_cfg,
        data_cfg=args.data_cfg,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        backend=args.backend,
        save_dir=Path(args.save_dir) if args.save_dir else None,
        run_name=args.run_name,
        epochs=args.epochs,
        lr=args.lr,
        num_calibration_batches=args.num_calibration_batches,
        quantize_botnet=not args.skip_botnet_quant,
        quantize_coordatt=not args.skip_ca_quant,
        sensitive_modules=args.sensitive_modules,
    )

    print(f"\n✓ Best model: {results['best_path']}")
    print(f"  Best mAP: {results['best_map']:.4f} at epoch {results['best_epoch']}")


if __name__ == "__main__":
    main()