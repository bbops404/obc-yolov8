"""Hybrid Quantization: PTQ + Targeted QAT Fine-tuning.

This script loads a PTQ-calibrated model and applies Quantization-Aware Training
(QAT) fine-tuning to sensitive modules only (BoTNet, CoordAtt). This is a short
fine-tuning phase (1-5 epochs) that recovers accuracy lost during PTQ.
"""

from __future__ import annotations
import torch
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
from torch.backends import quantized as torch_quantized_backends
from tqdm import tqdm




# Ensure the local ultralytics package (vendored in this repo) is importable.
REPO_ROOT = Path(__file__).parent
# Try obc-yolov8 path first, fallback to ultralytics10.24
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if not ULTRALYTICS_PATH.exists():
    ULTRALYTICS_PATH = REPO_ROOT / "ultralytics10.24"
import sys

if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO, __version__  # type: ignore  # noqa: E402
from ultralytics.nn import tasks as ultralytics_tasks  # type: ignore  # noqa: E402
from ultralytics.utils import LOGGER  # type: ignore  # noqa: E402
from ultralytics.utils.torch_utils import de_parallel  # type: ignore  # noqa: E402

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


def print_trainable_parameters(model, logger=None):
    """Print trainable vs frozen parameters in the model."""
    if logger is None:
        logger = LOGGER
    
    trainable_params = 0
    frozen_params = 0
    trainable_modules = []
    
    for name, module in model.named_modules():
        module_trainable = 0
        module_frozen = 0
        
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                module_trainable += param.numel()
                trainable_params += param.numel()
            else:
                module_frozen += param.numel()
                frozen_params += param.numel()
        
        if module_trainable > 0:
            trainable_modules.append((name, module_trainable, module_frozen))
    
    total_params = trainable_params + frozen_params
    
    logger.info("=" * 80)
    logger.info("TRAINABLE PARAMETERS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    
    if trainable_modules:
        logger.info(f"\nTrainable modules ({len(trainable_modules)} modules):")
        logger.info("-" * 80)
        for name, trainable, frozen in trainable_modules:
            total = trainable + frozen
            pct = 100 * trainable / total if total > 0 else 0
            logger.info(f"  {name:50s}  Trainable: {trainable:>10,} / {total:>10,} ({pct:>5.1f}%)")
    
    logger.info("=" * 80)


def freeze_all_parameters(model):
    """Freeze all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_sensitive_modules(model, sensitive_modules: List[str], logger=None):
    """Unfreeze only the specified sensitive modules.
    
    Args:
        model: The model to modify
        sensitive_modules: List of module name patterns to unfreeze (e.g., ['botnet', 'coordatt'])
        logger: Logger instance
    
    Returns:
        List of trainable parameters
    """
    if logger is None:
        logger = LOGGER
    
    trainable_params = []
    unfrozen_modules = []
    
    for name, module in model.named_modules():
        # Check if this module matches any sensitive pattern
        is_sensitive = any(
            pattern.lower() in name.lower() 
            for pattern in sensitive_modules
        )
        
        if is_sensitive:
            logger.info(f"Unfreezing sensitive module: {name}")
            unfrozen_modules.append(name)
            
            # Unfreeze all parameters in this module
            for param in module.parameters():
                param.requires_grad = True
                trainable_params.append(param)
    
    if not unfrozen_modules:
        logger.warning(f"⚠️  No modules matched patterns: {sensitive_modules}")
        logger.warning("    Model will have NO trainable parameters!")
    
    return trainable_params


def train_hybrid_qat(
    ptq_weights: str | Path,
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
    warmup_epochs: int = 0,
    sensitive_modules: Optional[List[str]] = None,
    evaluate_before: bool = True,
    evaluate_after: bool = True,
    save_best: bool = True,
    patience: int = 10,  # Early stopping patience
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Hybrid QAT: Fine-tune PTQ model on sensitive modules only.

    Args:
        ptq_weights: Path to PTQ-calibrated model weights (int8.pt from train_ptq.py)
        model_cfg: Path to model YAML configuration
        data_cfg: Path to dataset YAML configuration
        imgsz: Image size for training
        batch: Batch size for training
        workers: Number of data loading workers
        device: Device to use (GPU ID or 'cpu')
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        save_dir: Directory to save outputs
        run_name: Name for the run
        epochs: Number of fine-tuning epochs (default: 5)
        lr: Learning rate (default: 1e-4, much lower than full training)
        momentum: SGD momentum (default: 0.9)
        weight_decay: Weight decay (default: 1e-4)
        warmup_epochs: Number of warmup epochs (default: 0)
        sensitive_modules: List of module patterns to fine-tune (default: ['model.10', 'model.19', 'model.20', 'model.23', 'model.24'])
        evaluate_before: Evaluate PTQ model before fine-tuning
        evaluate_after: Evaluate after each epoch
        save_best: Save best model based on mAP
        patience: Early stopping patience (epochs without improvement)
        **kwargs: Additional arguments

    Returns:
        Dictionary with paths to saved models and metrics
    """

    project_dir = Path(save_dir) if save_dir is not None else DEFAULT_PROJECT
    run_name = run_name or DEFAULT_NAME
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Default sensitive modules (BoTNet and CoordAtt)
    if sensitive_modules is None:
        sensitive_modules = ['model.10', 'model.19', 'model.20', 'model.23', 'model.24']  # BoTNet + CoordAtt indices
    
    LOGGER.info("=" * 80)
    LOGGER.info("HYBRID QAT: PTQ + Targeted Fine-tuning")
    LOGGER.info("=" * 80)
    LOGGER.info(f"PTQ weights:      {ptq_weights}")
    LOGGER.info(f"Sensitive modules: {', '.join(sensitive_modules)}")
    LOGGER.info(f"Epochs:           {epochs}")
    LOGGER.info(f"Learning rate:    {lr}")
    LOGGER.info(f"Backend:          {backend}")
    LOGGER.info("=" * 80)

    # Load PTQ model
    LOGGER.info(f"\nLoading PTQ model from {ptq_weights}...")
    ptq_weights_path = Path(ptq_weights)
    if not ptq_weights_path.exists():
        raise FileNotFoundError(f"PTQ weights file not found: {ptq_weights_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ptq_weights_path, map_location='cpu')
    
    # Check if this is a PTQ checkpoint
    if not checkpoint.get('ptq', False):
        LOGGER.warning("⚠️  Warning: Checkpoint doesn't have 'ptq' flag. Are you sure this is a PTQ model?")
    
    # Load model - build a base YOLO model from the config and then attach
    # or load weights depending on what the checkpoint contains. Be explicit
    # about inner-model types so static type checkers (Pylance) don't warn
    # about calling `load_state_dict` on `None` or `str`.
    base_model = YOLO(str(model_cfg))

    # Helper: get the checkpoint entries (prefer 'model' then 'model_state_dict')
    loaded_model = checkpoint.get('model', None)
    if loaded_model is not None:
        LOGGER.info("Loading full model from checkpoint...")
        # The checkpoint 'model' entry may be an nn.Module, a state dict, or a path
        if isinstance(loaded_model, nn.Module):
            base_model.model = loaded_model
        elif isinstance(loaded_model, dict):
            inner = getattr(base_model, 'model', None)
            if not isinstance(inner, nn.Module):
                raise RuntimeError('YOLO did not create an inner model to load state_dict into')
            inner.load_state_dict(loaded_model)
        elif isinstance(loaded_model, (str, Path)):
            # The checkpoint stored a path/config string for the model; re-instantiate
            base_model = YOLO(model=str(loaded_model))
        else:
            raise TypeError(f"Unsupported type for checkpoint['model']: {type(loaded_model)!r}")
        model = base_model
    else:
        # Fallback to older-style `model_state_dict` key
        state = checkpoint.get('model_state_dict', None)
        if state is None:
            raise ValueError("Checkpoint doesn't contain 'model' or 'model_state_dict'")
        LOGGER.info("Loading model from state_dict...")
        if not isinstance(state, dict):
            raise TypeError("checkpoint['model_state_dict'] must be a state dict (dict)")
        inner = getattr(base_model, 'model', None)
        if not isinstance(inner, nn.Module):
            raise RuntimeError('YOLO did not create an inner model to load state_dict into')
        inner.load_state_dict(state, strict=False) 
        model = base_model

    detection_model = model.model
    # Sanity check: ensure inner detection model exists and is an nn.Module
    if not isinstance(detection_model, nn.Module):
        raise RuntimeError(f"Loaded checkpoint did not yield a valid nn.Module for detection_model (got {type(detection_model)!r})")
    LOGGER.info(f"✓ PTQ model loaded successfully")
# Set quantization backend EARLY
    # The block below is intentionally commented out to prevent PyTorch from
    # setting a global engine that lacks the necessary QAT derivatives (STE) 
    # required for the backward pass on non-standard architectures like M1.
    # if backend in torch_quantized_backends.supported_engines:
    #     torch_quantized_backends.engine = backend 
    #     LOGGER.info(f"Set quantization backend engine to {backend}")
    # else:
    #     LOGGER.warning(f"Backend '{backend}' not supported, using default: {torch_quantized_backends.engine}")
    LOGGER.warning("Ignoring explicit backend engine setting. Relying on default PyTorch CPU QAT path for M1 stability.")

    # Move model to device
    device_str = _resolve_device(device)
    
    # 🎯 FIX START 🎯
    # Determine the final PyTorch device object based on the resolved string
    if device_str == "cpu":
        # Force CPU if specified, which is required for stable QAT on M1/ARM
        torch_device = torch.device("cpu")
        LOGGER.info("Device explicitly set to CPU for stable QAT execution.")
    elif device_str.isdigit() or device_str.startswith("cuda:"):
        # Handle CUDA devices if available
        if torch.cuda.is_available():
            torch_device = torch.device(device_str)
        else:
            # Fallback for CUDA when unavailable
            torch_device = torch.device("cpu")
            device_str = "cpu"
            LOGGER.warning(f"CUDA device '{device_str}' requested but not available. Falling back to CPU.")
    else:
        # Default fallback to CPU for safety (e.g., if 'mps' was passed)
        torch_device = torch.device("cpu")
        device_str = "cpu"
        LOGGER.warning(f"Unsupported device '{_resolve_device(device)}'. Falling back to CPU for QAT stability.")

    # 🎯 FIX END 🎯
    
    model.to(torch_device)
    LOGGER.info(f"Model moved to device: {device_str}")
    # Evaluate PTQ baseline (before fine-tuning)
    ptq_map = None
    if evaluate_before:
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("Evaluating PTQ baseline (before fine-tuning)...")
        LOGGER.info("=" * 80)
        try:
            eval_results = model.val(
                data=str(data_cfg),
                imgsz=imgsz,
                batch=batch or 1,
                device=device_str,
                plots=False,
                save=False,
                verbose=True
            )
            
            if eval_results:
                ptq_map = getattr(eval_results, 'map', None) or getattr(eval_results, 'metrics', {}).get('map', None)
                map50 = getattr(eval_results, 'map50', None) or getattr(eval_results, 'metrics', {}).get('map50', None)
                
                LOGGER.info("\nPTQ Baseline Metrics:")
                if map50 is not None:
                    LOGGER.info(f"  mAP@0.5:      {map50:.4f} ({map50*100:.2f}%)")
                if ptq_map is not None:
                    LOGGER.info(f"  mAP@0.5:0.95: {ptq_map:.4f} ({ptq_map*100:.2f}%)")
                    LOGGER.info(f"\nTarget: Recover at least 2-3% mAP through fine-tuning")
                    LOGGER.info(f"Expected final mAP: ~{(ptq_map + 0.025):.4f} ({(ptq_map + 0.025)*100:.2f}%)")
        except Exception as e:
            LOGGER.warning(f"PTQ baseline evaluation failed: {e}")
            LOGGER.info("Continuing with fine-tuning anyway...")

    # Prepare model for QAT fine-tuning
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Preparing model for QAT fine-tuning...")
    LOGGER.info("=" * 80)
    
    # Put model in training mode
    detection_model.train()
    
    # Disable observers (keep PTQ scales fixed)
    LOGGER.info("Freezing PTQ observers (scales remain constant)...")
    detection_model.apply(torch.quantization.disable_observer)
    
    # Freeze ALL parameters first
    LOGGER.info("Freezing all parameters...")
    freeze_all_parameters(detection_model)
    
    # Unfreeze ONLY sensitive modules
    LOGGER.info("Unfreezing sensitive modules: %s" % ', '.join(sensitive_modules))
# --- START: FIX ---

# Use torch.enable_grad() to explicitly allow requires_grad=True calls
    with torch.enable_grad():
        trainable_params = unfreeze_sensitive_modules(detection_model, sensitive_modules, logger=LOGGER)
# --- END: FIX ---

    LOGGER.info(f"Total trainable parameters (unfrozen): {trainable_params}")
# ...
    if not trainable_params:
        raise ValueError("No trainable parameters found! Check sensitive_modules patterns.")
    
    # Print parameter summary
    print_trainable_parameters(detection_model, logger=LOGGER)
# Print parameter summary

  # train_hybrid_qat.py (Around the trainable_params section, line ~380)

    # ... (code to print parameter summary) ...
    print_trainable_parameters(detection_model, logger=LOGGER)
    
    # ====================================================================
    # 🎯 FIX: Correctly Attach cfg_args and Initialize Criterion 🎯
    # ====================================================================
    try:
        from ultralytics.cfg import get_cfg
        from ultralytics.utils.loss import v8DetectionLoss
        
        cfg_args = get_cfg(overrides={"imgsz": imgsz, "batch": batch or 16, "workers": workers or 8, "device": _resolve_device(device), "task": "detect"})
        
        setattr(model, 'args', cfg_args)
        setattr(detection_model, 'hyp', cfg_args) # Crucial: Provides .box, .cls, etc.
        setattr(detection_model, 'criterion', v8DetectionLoss(detection_model))
        LOGGER.info("✓ Successfully initialized model criterion and hyperparameters.")
        
    except Exception as e:
        LOGGER.error(f"FATAL: Failed to initialize loss criterion: {e}")
        import traceback
        traceback.print_exc()
        raise

    # ====================================================================

    # Setup optimizer... (continues here)

    # Setup optimizer (only for trainable parameters)
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Setting up optimizer...")
    LOGGER.info("=" * 80)
    
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    LOGGER.info(f"Optimizer: SGD")
    LOGGER.info(f"  Learning rate: {lr}")
    LOGGER.info(f"  Momentum:      {momentum}")
    LOGGER.info(f"  Weight decay:  {weight_decay}")
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.1  # Minimum LR is 10% of initial
    )
    
    LOGGER.info(f"Scheduler: CosineAnnealingLR (T_max={epochs})")

    # Load training data
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"Loading training data from {data_cfg}...")
    LOGGER.info("=" * 80)
    
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.cfg import get_cfg
    
    dataset = check_det_dataset(str(data_cfg))
    
    # Build training dataset
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
        shuffle=True,  # Shuffle for training
        rank=-1,
    )
    LOGGER.info(f"Training dataset: {len(train_loader)} batches")
    
    # Build validation dataset
    val_dataset = build_yolo_dataset(
        cfg=cfg_args,
        img_path=dataset.get("val", ""),
        batch=batch or 16,
        data=dataset,
        mode="val",
        rect=False,
        stride=32,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch=batch or 16,
        workers=workers or 8,
        shuffle=False,
        rank=-1,
    )
    LOGGER.info(f"Validation dataset: {len(val_loader)} batches")

    # Setup save directory
    save_dir_path = project_dir / run_name
    save_dir_path.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Ensure `last_path` is always defined for static checkers and callers
    last_path: Optional[Path] = None

    # Training loop
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"Starting QAT fine-tuning for {epochs} epochs...")
    LOGGER.info("=" * 80)
    
    best_map = ptq_map if ptq_map is not None else 0.0
    best_epoch = -1
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Epoch {epoch+1}/{epochs}")
        LOGGER.info(f"{'='*80}")
        
        # Training phase
        detection_model.train()
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        for batch_idx, batch_data in enumerate(pbar):
            # Move batch data to device
            images = batch_data['img'].to(torch_device, non_blocking=True).float() / 255.0
            
            # Move other batch data to device if needed
            if 'cls' in batch_data:
                batch_data['cls'] = batch_data['cls'].to(torch_device)
            if 'bboxes' in batch_data:
                batch_data['bboxes'] = batch_data['bboxes'].to(torch_device)
            if 'batch_idx' in batch_data:
                batch_data['batch_idx'] = batch_data['batch_idx'].to(torch_device)
            
            # Forward pass with fake quantization
            try:
                # YOLO forward pass
                outputs = detection_model(images)
                
                # Compute loss using YOLO's built-in loss function
                loss = compute_yolo_loss(outputs, batch_data, detection_model)
                
                # Check for invalid loss
                if not torch.isfinite(loss):
                    LOGGER.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                    continue
                
                # Backward pass (only sensitive layers get gradients)
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{running_loss/num_batches:.4f}'
                })
                
            except Exception as e:
                LOGGER.warning(f"Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        LOGGER.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        LOGGER.info(f"Learning rate: {current_lr:.6f} -> {new_lr:.6f}")
        
        # Evaluation phase
        if evaluate_after:
            LOGGER.info(f"\nEvaluating epoch {epoch+1}...")
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
                    current_map = getattr(eval_results, 'map', None) or getattr(eval_results, 'metrics', {}).get('map', None)
                    map50 = getattr(eval_results, 'map50', None) or getattr(eval_results, 'metrics', {}).get('map50', None)
                    
                    LOGGER.info(f"\nEpoch {epoch+1} Results:")
                    if map50 is not None:
                        LOGGER.info(f"  mAP@0.5:      {map50:.4f} ({map50*100:.2f}%)")
                    if current_map is not None:
                        LOGGER.info(f"  mAP@0.5:0.95: {current_map:.4f} ({current_map*100:.2f}%)")
                        
                        # Check improvement
                        if ptq_map is not None:
                            improvement = (current_map - ptq_map) * 100
                            LOGGER.info(f"  Improvement:  {improvement:+.2f}% from PTQ baseline")
                        
                        # Save best model
                        if current_map > best_map:
                            best_map = current_map
                            best_epoch = epoch + 1
                            epochs_without_improvement = 0
                            
                            if save_best:
                                best_path = weights_dir / "best.pt"
                                LOGGER.info(f"  ✓ New best mAP! Saving to {best_path}")
                                torch.save({
                                    "model": detection_model,
                                    "model_state_dict": detection_model.state_dict(),
                                    "epoch": epoch + 1,
                                    "best_fitness": current_map,
                                    "optimizer": optimizer.state_dict(),
                                    "date": datetime.now().isoformat(),
                                    "ptq": False,
                                    "hybrid_qat": True,
                                    "backend": backend,
                                }, best_path)
                        else:
                            epochs_without_improvement += 1
                            LOGGER.info(f"  No improvement for {epochs_without_improvement} epoch(s)")
                            
                            # Early stopping
                            if epochs_without_improvement >= patience:
                                LOGGER.info(f"\n⚠️  Early stopping triggered (patience={patience})")
                                LOGGER.info(f"   Best mAP: {best_map:.4f} at epoch {best_epoch}")
                                break
                    
            except Exception as e:
                LOGGER.warning(f"Evaluation failed: {e}")
        
        # Save last checkpoint
        last_path = weights_dir / "last.pt"
        torch.save({
            "model": detection_model,
            "model_state_dict": detection_model.state_dict(),
            "epoch": epoch + 1,
            "best_fitness": best_map,
            "optimizer": optimizer.state_dict(),
            "date": datetime.now().isoformat(),
            "ptq": False,
            "hybrid_qat": True,
            "backend": backend,
        }, last_path)

    # Training complete
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("QAT Fine-tuning Complete!")
    LOGGER.info("=" * 80)
    if ptq_map is not None and best_map > 0:
        improvement = (best_map - ptq_map) * 100
        LOGGER.info(f"PTQ baseline:  {ptq_map:.4f} ({ptq_map*100:.2f}%)")
        LOGGER.info(f"Best QAT:      {best_map:.4f} ({best_map*100:.2f}%)")
        LOGGER.info(f"Improvement:   {improvement:+.2f}%")
        LOGGER.info(f"Best epoch:    {best_epoch}")
    
    if save_best:
        LOGGER.info(f"\nBest model saved to: {weights_dir / 'best.pt'}")
    LOGGER.info(f"Last model saved to: {weights_dir / 'last.pt'}")
    LOGGER.info("=" * 80)

    return {
        "best_path": weights_dir / "best.pt" if save_best else None,
        "last_path": last_path,
        "weights_dir": weights_dir,
        "best_map": best_map,
        "best_epoch": best_epoch,
    }


# train_hybrid_qat.py, function compute_yolo_loss (around line 660)

# train_hybrid_qat.py (The function that calculates the loss)

def compute_yolo_loss(outputs, batch_data, model):
    """Compute YOLO detection loss using model's built-in loss function."""

    # We must ensure the loss criterion has the hyperparameters (hyp) object.
    criterion = model.criterion
    
    # 1. Manually retrieve the hyperparameters object we attached earlier.
    # We attached 'cfg_args' as 'hyp' to the detection_model in the setup.
    hyp_object = getattr(model, 'hyp', None)
    
    if hyp_object is None:
        raise RuntimeError("Hyperparameters ('hyp') not found on detection model. QAT setup failed.")

    # 2. Check the criterion's 'hyp' attribute. If it's the default dict 
    # (which causes the error), re-assign it to our correct object.
    # This is necessary because the criterion might have been initialized 
    # with a dummy dict or an old version structure.
    if isinstance(criterion.hyp, dict):
        criterion.hyp = hyp_object 

    # 3. Call the loss criterion directly.
    # We call the model's loss method which uses the now-fixed criterion.
    if hasattr(model, 'loss'):
        loss, loss_items = model.loss(batch_data, outputs)
        return loss
    
    raise RuntimeError(f"Cannot compute YOLO loss. Detection model {type(model).__name__} lacks a 'loss' method.")
def main():
    """Command-line interface for Hybrid QAT."""
    parser = argparse.ArgumentParser(description="Hybrid QAT: PTQ + Targeted Fine-tuning for YOLOv8-CA")
    parser.add_argument(
        "--ptq-weights",
        type=str,
        required=True,
        help="Path to PTQ model weights (int8.pt from train_ptq.py)",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default=str(DEFAULT_MODEL_CFG),
        help="Path to model YAML configuration",
    )
    parser.add_argument(
        "--data-cfg",
        type=str,
        default=str(DEFAULT_DATA_CFG),
        help="Path to dataset YAML configuration",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (GPU ID or 'cpu')",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fbgemm",
        choices=["qnnpack", "fbgemm"],
        help="Quantization backend",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--sensitive-modules",
        type=str,
        nargs='+',
        default=None,
        help="Module patterns to fine-tune (default: BoTNet and CoordAtt)",
    )
    parser.add_argument(
        "--no-eval-before",
        action="store_true",
        help="Skip evaluation of PTQ baseline",
    )
    parser.add_argument(
        "--no-eval-after",
        action="store_true",
        help="Skip evaluation after each epoch",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10 epochs)",
    )

    args = parser.parse_args()

    results = train_hybrid_qat(
        ptq_weights=args.ptq_weights,
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
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        sensitive_modules=args.sensitive_modules,
        evaluate_before=not args.no_eval_before,
        evaluate_after=not args.no_eval_after,
        patience=args.patience,
    )

    print("\n" + "=" * 80)
    print("Hybrid QAT Complete!")
    print("=" * 80)
    if isinstance(results, dict):
        print("Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        print(f"Results: {results}")


if __name__ == "__main__":
    main()
