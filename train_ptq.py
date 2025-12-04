"""Post-Training Quantization (PTQ) entry point.

This script exposes a `train_ptq` helper that prepares the YOLOv8-CA
architecture for PTQ, calibrates it with calibration data, and converts
it to an INT8 checkpoint. No training is required for PTQ.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
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
DEFAULT_NAME = "train_ptq"
DEFAULT_MODEL_CFG = ULTRALYTICS_PATH / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-CA.yaml"
DEFAULT_DATA_CFG = ULTRALYTICS_PATH / "ultralytics" / "cfg" / "datasets" / "combined_china_motorbike.yaml"


def _resolve_device(device: Any) -> str:
    """Convert device to string format."""
    if isinstance(device, torch.device):
        return str(device)
    if isinstance(device, (list, tuple)):
        return ",".join(map(str, device))
    return str(device)


def print_quantized_layers(model, logger=None):
    """
    Print all quantized layers in the model.
    
    Args:
        model: The quantized model to inspect
        logger: Optional logger instance (defaults to LOGGER)
    """
    if logger is None:
        logger = LOGGER
    
    import torch.nn as nn
    from ultralytics.nn.ODConv import ODConv
    
    # Collect all quantized layers with their values
    quantized_layers = []
    odconv_layers = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_path = type(module).__module__
        
        # Track ODConv layers to verify they're not quantized
        if isinstance(module, ODConv):
            odconv_layers.append(name)
        
        # Check if it's a quantized Conv2d directly (not wrapped)
        # Quantized Conv2d modules are from torch.ao.nn.quantized, not torch.nn
        if module_type == 'Conv2d' and 'quantized' in module_path.lower():
            if not name.endswith('.conv'):  # Skip .conv children (handled separately)
                scale = None
                zero_point = None
                if hasattr(module, 'scale'):
                    scale = module.scale.item() if isinstance(module.scale, torch.Tensor) else module.scale
                if hasattr(module, 'zero_point'):
                    zero_point = module.zero_point.item() if isinstance(module.zero_point, torch.Tensor) else module.zero_point
                quantized_layers.append((name, 'Conv2d', scale, zero_point))
        
        # Check inside Conv wrappers
        if hasattr(module, 'conv') and isinstance(module.conv, nn.Module):
            conv_module = module.conv
            conv_path = type(conv_module).__module__
            conv_type = type(conv_module).__name__
            is_quantized = (
                hasattr(conv_module, '_packed_params') or 
                'quantized' in conv_path.lower() or
                'Quantized' in conv_type
            )
            if is_quantized:
                scale = None
                zero_point = None
                if hasattr(conv_module, 'scale'):
                    scale = conv_module.scale.item() if isinstance(conv_module.scale, torch.Tensor) else conv_module.scale
                if hasattr(conv_module, 'zero_point'):
                    zero_point = conv_module.zero_point.item() if isinstance(conv_module.zero_point, torch.Tensor) else conv_module.zero_point
                quantized_layers.append((f'{name}.conv', 'Conv2d (wrapped)', scale, zero_point))
        
        # Check for quantized Linear layers
        if module_type == 'Linear' and 'quantized' in module_path.lower():
            scale = None
            zero_point = None
            if hasattr(module, 'scale'):
                scale = module.scale.item() if isinstance(module.scale, torch.Tensor) else module.scale
            if hasattr(module, 'zero_point'):
                zero_point = module.zero_point.item() if isinstance(module.zero_point, torch.Tensor) else module.zero_point
            quantized_layers.append((name, 'Linear', scale, zero_point))
    
    # Remove duplicates and sort by layer name
    seen = set()
    unique_layers = []
    for layer_name, layer_type, scale, zero_point in quantized_layers:
        if layer_name not in seen:
            seen.add(layer_name)
            unique_layers.append((layer_name, layer_type, scale, zero_point))
    quantized_layers = sorted(unique_layers, key=lambda x: x[0])
    
    # Print summary
    logger.info("=" * 80)
    logger.info("QUANTIZED LAYERS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total quantized layers: {len(quantized_layers)}")
    logger.info(f"ODConv layers (should NOT be quantized): {len(odconv_layers)}")
    
    if odconv_layers:
        logger.info("\nODConv layers found (verify these are NOT quantized):")
        for odconv_name in odconv_layers:
            logger.info(f"  - {odconv_name}")
    
    if quantized_layers:
        logger.info(f"\nQuantized layers ({len(quantized_layers)} total):")
        logger.info("-" * 80)
        
        # Group by type
        conv2d_layers = [l for l in quantized_layers if 'Conv2d' in l[1]]
        linear_layers = [l for l in quantized_layers if l[1] == 'Linear']
        
        logger.info(f"\nConv2d layers ({len(conv2d_layers)}):")
        for i, (layer_name, layer_type, scale, zp) in enumerate(conv2d_layers, 1):
            scale_str = f'{scale:.8f}' if scale is not None else 'N/A'
            zp_str = f'{zp}' if zp is not None else 'N/A'
            logger.info(f"  {i:3d}. {layer_name:50s}  [{layer_type:20s}]  scale={scale_str:12s}  zp={zp_str:4s}")
        
        if linear_layers:
            logger.info(f"\nLinear layers ({len(linear_layers)}):")
            for i, (layer_name, layer_type, scale, zp) in enumerate(linear_layers, 1):
                scale_str = f'{scale:.8f}' if scale is not None else 'N/A'
                zp_str = f'{zp}' if zp is not None else 'N/A'
                logger.info(f"  {i:3d}. {layer_name:50s}  [{layer_type:20s}]  scale={scale_str:12s}  zp={zp_str:4s}")

                # --- ABILITY MODULE VERIFICATION ---
        botnet_quantized = any(name.startswith('model.10.') for name, _, _, _ in quantized_layers)
        coordatt_quantized = any(name.startswith('model.20.') or name.startswith('model.24.') for name, _, _, _ in quantized_layers)
        
        logger.info("\n--- Ablation Module Status ---")
        
        if botnet_quantized:
            logger.warning("⚠️  BoTNet (model.10) layers WERE FOUND in the quantized list!")
        else:
            logger.info("✓ BoTNet (model.10) layers successfully EXCLUDED (FP32).")
            
        if coordatt_quantized:
            logger.warning("⚠️  CoordAtt (model.20/24) layers WERE FOUND in the quantized list!")
        else:
            logger.info("✓ CoordAtt (model.20/24) layers successfully EXCLUDED (FP32).")
        
        # --- END ABILITY MODULE VERIFICATION ---
        # Verify no ODConv wrapper layers are in the quantized list
        # Only check for exact matches, not substring matches (child layers are OK)
        odconv_in_quantized = [name for name, _, _, _ in quantized_layers if name in odconv_layers]
        if odconv_in_quantized:
            logger.warning(f"\n⚠️  WARNING: Found {len(odconv_in_quantized)} ODConv wrapper layers in quantized list!")
            for name in odconv_in_quantized:
                logger.warning(f"  - {name}")
        else:
            logger.info("\n✓ Verified: No ODConv wrapper layers found in quantized list")
            if odconv_layers:
                logger.info(f"  Note: ODConv internal attention layers may be quantized (this is expected)")
    else:
        logger.warning("⚠️  No quantized layers found! Model may not have been properly quantized.")
    
    logger.info("=" * 80)


def train_ptq(
    weights: str | Path,
    model_cfg: str | Path = DEFAULT_MODEL_CFG,
    data_cfg: str | Path = DEFAULT_DATA_CFG,
    imgsz: int = 640,
    batch: Optional[int] = None,
    workers: Optional[int] = None,
    device: Any = 0,
    backend: str = "qnnpack",
    save_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    convert_to_int8: bool = True,
    use_fx: bool = True,
    num_calibration_batches: Optional[int] = None,
    calibration_split: str = "val",  # 'val' or 'train'
    evaluate: bool = False,  # Skip evaluation by default (can cause segfaults with quantized models)

    quantize_botnet: bool = True,
    quantize_coordatt: bool = True,
    **kwargs: Any,
) -> Dict[str, Optional[Path]]:
    """Run PTQ calibration and optionally export an INT8 model.

    Args:
        model_cfg: Path to model YAML configuration
        data_cfg: Path to dataset YAML configuration
        weights: Path to pretrained FP32 weights (required for PTQ)
        imgsz: Image size for inference
        batch: Batch size for calibration
        workers: Number of data loading workers
        device: Device to use (GPU ID or 'cpu')
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        save_dir: Directory to save outputs
        run_name: Name for the run
        convert_to_int8: Whether to convert to INT8 after calibration
        use_fx: Whether to use FX mode for quantization
        num_calibration_batches: Number of batches to use for calibration (None = all)
        calibration_split: Which split to use for calibration ('val' or 'train')
        **kwargs: Additional arguments

    Returns:
        Dictionary with keys `int8_path` and `weights_dir`. Missing artifacts are None.
    """

    project_dir = Path(save_dir) if save_dir is not None else DEFAULT_PROJECT
    run_name = run_name or DEFAULT_NAME
    project_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained FP32 model
    LOGGER.info("Loading pretrained FP32 model...")
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    if weights_path.is_dir():
        raise ValueError(f"Weights path must be a file, not a directory: {weights_path}")
    
    if not weights_path.suffix == '.pt':
        LOGGER.warning(f"Weights file does not have .pt extension: {weights_path}")
    
    model = YOLO(str(weights_path))
    LOGGER.info(f"Loaded model from {weights_path}")

    detection_model = getattr(model, "model", None)
    if detection_model is None:
        LOGGER.warning("Loaded checkpoint did not expose a DetectionModel. Rebuilding from model config...")
        model = YOLO(str(model_cfg))
        model.load(str(weights_path))
        detection_model = getattr(model, "model", None)

    if detection_model is None:
        raise RuntimeError(
            "Failed to build a DetectionModel from the provided weights/config. "
            "Ensure the weights file is a training checkpoint (.pt) and not an exported runtime."
        )

    if not hasattr(detection_model, "prepare_for_ptq"):
        model_type = type(detection_model).__name__
        raise AttributeError(
            f"Loaded model (type: {model_type}) does not implement prepare_for_ptq(). "
            f"This method is only available for DetectionModel. "
            f"Make sure you're loading a detection model checkpoint."
        )

    # Prepare model for PTQ
    example_input = torch.randn(1, detection_model.yaml.get("ch", 3), imgsz, imgsz)
    LOGGER.info(f"Preparing model for PTQ with backend '{backend}' and image size {imgsz}")
    
    # Get quantization configuration from kwargs (defaults to all True)
    quantize_backbone = kwargs.get('quantize_backbone', True)  # Still use kwargs for others
    quantize_neck = kwargs.get('quantize_neck', True)
    
    quantize_parts = []
    if quantize_backbone:
        quantize_parts.append("backbone")
    if quantize_neck:
        quantize_parts.append("neck")
    if quantize_botnet:
        quantize_parts.append("BoTNet")
    if quantize_coordatt:
        quantize_parts.append("CoordAtt")
    
    if quantize_parts:
        LOGGER.info(f"Selective quantization: {', '.join(quantize_parts)} (ODConv excluded)")
    else:
        LOGGER.warning("⚠️  No components selected for quantization!")
    
    prepared_model = detection_model.prepare_for_ptq(
        backend=backend,
        example_input=example_input,
        use_fx=use_fx,
        quantize_backbone=quantize_backbone,
        quantize_neck=quantize_neck,
        quantize_botnet=quantize_botnet,      # Uses the new flag
        quantize_coordatt=quantize_coordatt,  # Uses the new flag
    )
    model.model = prepared_model

    # We must operate on the prepared_model object to find modules by index
    model_for_manual_skip = model.model

    if not quantize_botnet:
        LOGGER.info("!!! MANUAL SKIP: Excluding BoTNet (model.10) from INT8 quantization.")
        try:
            # 1. Access the BoTNet module
            layer_list = model_for_manual_skip.model # Access the Sequential module
            botnet_module = layer_list.get_submodule('10') 
            
            # 2. Recursively clear qconfig for the BoTNet module and all its children
            cleared_count = 0
            
            # Set parent qconfig to None (in case it matters for the overall conversion)
            if hasattr(botnet_module, 'qconfig'):
                botnet_module.qconfig = None
            
            # Iterate through all named sub-modules and clear their qconfig
            for name, module in botnet_module.named_modules():
                if hasattr(module, 'qconfig') and module.qconfig is not None:
                    module.qconfig = None
                    cleared_count += 1
            
            if cleared_count > 0:
                LOGGER.info(f"!!! BoTNet (model.10) qconfig successfully cleared for {cleared_count} sub-modules (FP32 retention).")
            else:
                LOGGER.warning("!!! BoTNet module (model.10) found, but no qconfig attributes needed clearing.")

        except AttributeError as e:
            LOGGER.error(f"!!! Could not find module at path '10' (BoTNet) for manual skip. Error: {e}")


    if not quantize_coordatt:
        LOGGER.info("!!! MANUAL SKIP: Excluding CoordAtt (model.20 and model.24) from INT8 quantization.")
        layer_list = model_for_manual_skip.model # Access the Sequential module
        for index_str in ['19','23','20', '24']: 
            try:
                ca_module = layer_list.get_submodule(index_str)
                cleared_count = 0
                
                # Recursively clear qconfig for the CoordAtt module and all its children
                if hasattr(ca_module, 'qconfig'):
                    ca_module.qconfig = None
                
                for name, module in ca_module.named_modules():
                    if hasattr(module, 'qconfig') and module.qconfig is not None:
                        module.qconfig = None
                        cleared_count += 1
                
                if cleared_count > 0:
                     LOGGER.info(f"!!! CoordAtt (model.{index_str}) qconfig successfully cleared for {cleared_count} sub-modules (FP32 retention).")
                else:
                    LOGGER.warning(f"!!! CoordAtt module (model.{index_str}) found, but no qconfig attributes needed clearing.")
                    
            except AttributeError as e:
                LOGGER.error(f"!!! Could not find module at path '{index_str}' (CoordAtt) for manual skip. Error: {e}")

    # --- BoTNet partial skip: keep fc1 in FP32 even if BoTNet is quantized ---
    # Note: key, query, and value layers are now included in PTQ quantization
    botnet_skip_names = {
        "model.10.m.0.fc1",
    }
    botnet_cleared = 0
    for name, module in model_for_manual_skip.named_modules():
        if name in botnet_skip_names:
            if hasattr(module, "qconfig") and module.qconfig is not None:
                module.qconfig = None
                botnet_cleared += 1
            for child_name, child in module.named_modules():
                if hasattr(child, "qconfig") and child.qconfig is not None:
                    child.qconfig = None
                    botnet_cleared += 1
    if botnet_cleared > 0:
        LOGGER.info(f"!!! BoTNet attention fc1 qconfig cleared for {botnet_cleared} modules (kept FP32).")

    # --- ODConv SKIP: ensure ALL ODConv-related modules stay FP32 ---
    # The ODConv block is at index 1 in this architecture (model.1.*).
    # We clear qconfig both based on type (ODConv) and by name prefix 'model.1.'
    from ultralytics.nn.ODConv import ODConv
    LOGGER.info("!!! MANUAL SKIP: Excluding ALL ODConv-related modules from INT8 quantization (keep FP32).")
    odconv_cleared = 0
    for name, module in model_for_manual_skip.named_modules():
        is_odconv_block = isinstance(module, ODConv) or name.startswith("model.1.")
        if is_odconv_block:
            # Clear qconfig on this module
            if hasattr(module, "qconfig") and module.qconfig is not None:
                module.qconfig = None
                odconv_cleared += 1
            # Clear qconfig on all its children (internal attention convs, etc.)
            for child_name, child in module.named_modules():
                if hasattr(child, "qconfig") and child.qconfig is not None:
                    child.qconfig = None
                    odconv_cleared += 1
    if odconv_cleared > 0:
        LOGGER.info(f"!!! ODConv qconfig successfully cleared for {odconv_cleared} modules/sub-modules (ODConv kept in FP32).")
    else:
        LOGGER.info("ODConv-related modules found but no qconfig attributes needed clearing (already FP32).")

    # --- END: MANUAL PTQ SKIP IMPLEMENTATION ---

    # CRITICAL: Update detection_model to point to prepared_model so calibration works on the right model
    # The prepared model has observers, but we need to ensure calibrate_ptq works on it
    # Check if prepared_model has calibrate_ptq method (it should if it's still a DetectionModel)
    if hasattr(prepared_model, 'calibrate_ptq'):
        # Prepared model is still a DetectionModel, use it directly
        detection_model = prepared_model
        LOGGER.info("✓ Using prepared model for calibration (has calibrate_ptq method)")
    else:
        # Prepared model is wrapped or doesn't have calibrate_ptq
        # CRITICAL: We MUST use prepared_model for calibration because it has the observers!
        # Even if it doesn't have calibrate_ptq, we can call calibrate_ptq on the original
        # but we need to make sure the observers are in prepared_model
        LOGGER.warning("⚠️  Prepared model doesn't have calibrate_ptq method")
        LOGGER.warning("   This means prepare() returned a wrapped model")
        LOGGER.warning("   We'll use prepared_model for calibration to ensure observers are triggered")
        
        # Try to add calibrate_ptq to prepared_model by binding it from detection_model
        if hasattr(detection_model, 'calibrate_ptq'):
            import types
            prepared_model.calibrate_ptq = types.MethodType(detection_model.calibrate_ptq, prepared_model)
            detection_model = prepared_model
            LOGGER.info("✓ Bound calibrate_ptq method to prepared model")
        else:
            # Last resort: use prepared_model directly and hope observers work
            detection_model = prepared_model
            LOGGER.warning("   Using prepared_model directly - calibration may not work correctly")

    # Move model to device
    device_str = _resolve_device(device)
    
    # CRITICAL: For fbgemm/qnnpack backends, quantized operations MUST run on CPU
    # So we keep the model on CPU even if CUDA is requested
    if backend in ['qnnpack']:
        LOGGER.info(f"Backend {backend} requires CPU - keeping model on CPU throughout PTQ")
        torch_device = torch.device("cpu")
        device_str = "cpu"
    else:
        # Convert device string to torch.device
        if device_str == "cpu":
            torch_device = torch.device("cpu")
        elif device_str.isdigit() and torch.cuda.is_available():
            # Convert "0" -> "cuda:0"
            torch_device = torch.device(f"cuda:{device_str}")
            device_str = f"cuda:{device_str}"
        elif device_str.startswith("cuda:") and torch.cuda.is_available():
            torch_device = torch.device(device_str)
        else:
            torch_device = torch.device("cpu")
            device_str = "cpu"
    
    model.to(torch_device)
    LOGGER.info(f"Model moved to device: {device_str}")
    
    # CRITICAL: Set quantization backend engine EARLY, before any quantized operations
    # This must be set before prepare_for_ptq() creates any quantized infrastructure
    if backend in torch_quantized_backends.supported_engines:
        torch_quantized_backends.engine = backend
        LOGGER.info(f"Set quantization backend engine to {backend} (before PTQ preparation)")
    else:
        LOGGER.warning(f"Backend '{backend}' not supported, using default: {torch_quantized_backends.engine}")

    # Load calibration data
    LOGGER.info(f"Loading calibration data from {data_cfg} (split: {calibration_split})...")
    
    # Get data dictionary
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.cfg import get_cfg
    
    dataset = check_det_dataset(str(data_cfg))
    
    # Create proper config object using get_cfg
    cfg_args = get_cfg(overrides={
        "imgsz": imgsz,
        "batch": batch or 1,
        "workers": workers or 8,
        "device": device_str,
        "task": "detect",
    })
    
    if calibration_split == "val":
        # Use validation set for calibration
        val_dataset = build_yolo_dataset(
            cfg=cfg_args,
            img_path=dataset.get("val", ""),
            batch=batch or 1,
            data=dataset,
            mode="val",
            rect=False,
            stride=32,
        )
        calibration_dataloader = build_dataloader(
            dataset=val_dataset,
            batch=batch or 1,
            workers=workers or 8,
            shuffle=False,  # Don't shuffle for calibration
            rank=-1,
        )
        LOGGER.info(f"Using validation set for calibration ({len(calibration_dataloader)} batches)")
    else:
        # Use training set for calibration
        train_dataset = build_yolo_dataset(
            cfg=cfg_args,
            img_path=dataset.get("train", ""),
            batch=batch or 1,
            data=dataset,
            mode="train",
            rect=False,
            stride=32,
        )
        calibration_dataloader = build_dataloader(
            dataset=train_dataset,
            batch=batch or 1,
            workers=workers or 8,
            shuffle=False,  # Don't shuffle for calibration
            rank=-1,
        )
        LOGGER.info(f"Using training set for calibration ({len(calibration_dataloader)} batches)")

    # Calibrate model - CRITICAL: Use detection_model which now points to prepared_model
    LOGGER.info("Starting PTQ calibration...")
    if num_calibration_batches is not None:
        LOGGER.info(f"Using {num_calibration_batches} batches for calibration")
    else:
        LOGGER.info("Using all available batches for calibration")
    
    # CRITICAL: Use prepared_model (which has observers) for calibration, not detection_model
    # Even if detection_model was updated, ensure we're using the model with observers
    model_for_calibration = prepared_model if hasattr(prepared_model, 'calibrate_ptq') else detection_model
    
    # Verify we're using the right model
    from torch.ao.quantization import ObserverBase
    obs_count = sum(1 for name, module in model_for_calibration.named_modules() if isinstance(module, ObserverBase))
    activation_obs_count = sum(1 for name, module in model_for_calibration.named_modules() if hasattr(module, 'activation_post_process') and module.activation_post_process is not None)
    LOGGER.info(f"Model for calibration has {obs_count} ObserverBase modules and {activation_obs_count} activation_post_process observers")
    
    if obs_count == 0 and activation_obs_count == 0:
        LOGGER.error("⚠️  ERROR: Model for calibration has NO observers! This will cause calibration to fail!")
        LOGGER.error("   Using prepared_model instead...")
        model_for_calibration = prepared_model
    
    calibrated_model = model_for_calibration.calibrate_ptq(
        calibration_data=calibration_dataloader,
        num_batches=num_calibration_batches,
    )
    model.model = calibrated_model
    
    # --- START: Save Calibrated Model (can be used for QAT if needed) ---
    
    # Create necessary directories
    save_dir_path = project_dir / run_name
    save_dir_path.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Define the save path for the calibrated model
    # This calibrated model can optionally be used for QAT fine-tuning later
    calibrated_path = weights_dir / "calibrated.pt"
    LOGGER.info(f"Saving calibrated PTQ model to {calibrated_path}")
    
    # Save the calibrated model state dictionary
    # This state_dict includes the populated min/max statistics in the Observer modules
    torch.save({
        "model_state_dict": calibrated_model.state_dict(),
        "yaml": detection_model.yaml,
        "epoch": -1,  # PTQ doesn't have epochs
        "best_fitness": None,
        "date": datetime.now().isoformat(),
        "ptq": True,
        "backend": backend,
        "calibrated": True,  # Flag to identify this as a calibrated PTQ model
    }, calibrated_path)
    
    LOGGER.info(f"✓ Calibrated PTQ model saved to {calibrated_path}")
    # Convert to INT8
    int8_path = None
    weights_dir: Optional[Path] = None
    if convert_to_int8:
        LOGGER.info("Converting calibrated model to INT8...")
        # Pass backend to ensure it's set correctly during conversion
        int8_model = detection_model.convert_ptq_to_int8(calibrated_model=calibrated_model, backend=backend)
        model.model = int8_model

        # Save INT8 model
        save_dir_path = project_dir / run_name
        save_dir_path.mkdir(parents=True, exist_ok=True)
        weights_dir = save_dir_path / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        int8_path = weights_dir / "int8.pt"
        LOGGER.info(f"Saving INT8 model to {int8_path}")
        
        # Check if forward is a local function (can't be pickled)
        # If so, we'll save only state_dict to avoid pickling issues
        forward_is_local = hasattr(int8_model, 'forward') and (
            hasattr(int8_model.forward, '__qualname__') and 
            '<locals>' in str(int8_model.forward.__qualname__)
        )
        
        if forward_is_local:
            LOGGER.info("  Forward method is a local function - saving state_dict only to avoid pickling issues")
            # Save only state_dict and metadata (model can be reconstructed from yaml)
            torch.save({
                "model_state_dict": int8_model.state_dict(),
                "yaml": detection_model.yaml,
                "epoch": -1,  # PTQ doesn't have epochs
                "best_fitness": None,
                "date": datetime.now().isoformat(),
                "ptq": True,
                "backend": backend,
            }, int8_path)
        else:
            # Save model - save both state_dict and full model for flexibility
            # Full model object is needed for proper evaluation without segfaults
            torch.save({
                "model": int8_model,  # Save full model object (safer for evaluation)
                "model_state_dict": int8_model.state_dict(),  # Also save state_dict for loading flexibility
                "yaml": detection_model.yaml,
                "epoch": -1,  # PTQ doesn't have epochs
                "best_fitness": None,
                "date": datetime.now().isoformat(),
                "ptq": True,
                "backend": backend,
            }, int8_path)
        
        LOGGER.info(f"✓ INT8 model saved to {int8_path}")
        
        # Print quantized layers summary
        LOGGER.info("")
        print_quantized_layers(int8_model, logger=LOGGER)

    # Evaluate INT8 model if converted
    if convert_to_int8 and int8_path and evaluate:
        LOGGER.info("Evaluating INT8 model...")
        try:
            # CRITICAL: Set quantization backend engine BEFORE evaluation
            # This must match the backend used during conversion
            if backend in torch_quantized_backends.supported_engines:
                torch_quantized_backends.engine = backend
                LOGGER.info(f"Set quantization backend engine to {backend} for evaluation")
            else:
                LOGGER.warning(f"Backend '{backend}' not supported, using default: {torch_quantized_backends.engine}")
            
            # The model has already been repaired in convert_ptq_to_int8, so we can use it directly
            # But we need to ensure it's in eval mode and test forward pass
            LOGGER.info("Preparing model for evaluation...")
            int8_model = model.model
            
            # Ensure bookkeeping is still intact (should be from convert_ptq_to_int8)
            ensure_module_bookkeeping(int8_model, recursive=True)
            
            # Try to set eval mode (may fail for quantized models, but try anyway)
            try:
                int8_model.eval()
                LOGGER.info("Model set to eval mode")
            except (AttributeError, RuntimeError) as e:
                LOGGER.info(f"Model already in eval mode or quantized structure: {type(e).__name__}")
                # Quantized models may not support .eval() due to structure, but that's OK
            
            # Note: Forward pass test is skipped for quantized models
            # Quantized operations have special device requirements that are better handled
            # by the evaluation pipeline itself
            LOGGER.info("Model prepared for evaluation (bookkeeping verified)")
            
            # Run evaluation using YOLO's val method
            # For fbgemm/qnnpack backends, must use CPU
            eval_device = "cpu" if backend in ['qnnpack'] else device_str
            if eval_device != device_str:
                LOGGER.info(f"Using CPU for evaluation (required for {backend} backend)")
            
            LOGGER.info("Running validation with YOLO's val() method...")
            eval_results = model.val(
                data=str(data_cfg),
                imgsz=imgsz,
                batch=batch or 1,
                device=eval_device,
                plots=False,
                save=False,
                verbose=True
            )
            
            LOGGER.info("INT8 Model Evaluation Results:")
            if eval_results:
                # Extract metrics from results - handle both dict and DetMetrics object
                if isinstance(eval_results, dict):
                    map50 = eval_results.get('metrics/mAP50(B)', eval_results.get('map50', None))
                    map = eval_results.get('metrics/mAP50-95(B)', eval_results.get('map', None))
                    precision = eval_results.get('metrics/precision(B)', eval_results.get('precision', None))
                    recall = eval_results.get('metrics/recall(B)', eval_results.get('recall', None))
                else:
                    # It's a DetMetrics object - access attributes directly
                    map50 = getattr(eval_results, 'map50', getattr(eval_results, 'metrics', {}).get('map50', None) if hasattr(eval_results, 'metrics') else None)
                    map = getattr(eval_results, 'map', getattr(eval_results, 'metrics', {}).get('map', None) if hasattr(eval_results, 'metrics') else None)
                    precision = getattr(eval_results, 'precision', getattr(eval_results, 'metrics', {}).get('precision', None) if hasattr(eval_results, 'metrics') else None)
                    recall = getattr(eval_results, 'recall', getattr(eval_results, 'metrics', {}).get('recall', None) if hasattr(eval_results, 'metrics') else None)
                    
                    # Also try accessing via metrics attribute if it exists
                    if hasattr(eval_results, 'metrics'):
                        metrics = eval_results.metrics
                        if isinstance(metrics, dict):
                            map50 = metrics.get('map50', map50)
                            map = metrics.get('map', map)
                            precision = metrics.get('precision', precision)
                            recall = metrics.get('recall', recall)
                
                if map50 is not None:
                    LOGGER.info(f"  mAP@0.5:      {map50:.4f} ({map50*100:.2f}%)")
                if map is not None:
                    LOGGER.info(f"  mAP@0.5:0.95: {map:.4f} ({map*100:.2f}%)")
                if precision is not None:
                    LOGGER.info(f"  Precision:    {precision:.4f} ({precision*100:.2f}%)")
                if recall is not None:
                    LOGGER.info(f"  Recall:       {recall:.4f} ({recall*100:.2f}%)")
            else:
                LOGGER.warning("Evaluation completed but no metrics returned")
                
        except (RuntimeError, AttributeError, TypeError) as e:
            error_msg = str(e)
            if "segmentation" in error_msg.lower() or "core dumped" in error_msg.lower():
                LOGGER.warning("Evaluation caused segmentation fault")
                LOGGER.info("  The INT8 model was saved successfully and can be evaluated separately")
            else:
                LOGGER.warning(f"Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                LOGGER.info("  The INT8 model was saved successfully and can be evaluated separately")
        except Exception as e:
            LOGGER.warning(f"Evaluation failed with unexpected error: {e}")
            LOGGER.info("  The INT8 model was saved successfully and can be evaluated separately")
            import traceback
            traceback.print_exc()

    return {
        "int8_path": int8_path,
        "weights_dir": weights_dir if convert_to_int8 else None,
    }


def main():
    """Command-line interface for PTQ."""
    parser = argparse.ArgumentParser(description="Post-Training Quantization for YOLOv8-CA")
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
        "--weights",
        type=str,
        required=True,
        help="Path to pretrained FP32 weights (required)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size for calibration",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
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
        default="qnnpack",
        choices=["qnnpack"],
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
        "--no-convert",
        action="store_true",
        help="Skip INT8 conversion (only calibrate)",
    )
    parser.add_argument(
        "--no-fx",
        action="store_true",
        help="Disable FX mode (use eager mode)",
    )
    parser.add_argument(
        "--num-calibration-batches",
        type=int,
        default=None,
        help="Number of batches to use for calibration (None = all)",
    )
    parser.add_argument(
        "--calibration-split",
        type=str,
        default="val",
        choices=["val", "train"],
        help="Which split to use for calibration",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate INT8 model after conversion (may cause segfaults - use with caution)",
    )
    # --- ADD NEW ARGUMENTS FOR ABLATION STUDY ---
    parser.add_argument(
        "--skip-botnet-quant",
        action="store_true",
        help="Skip INT8 quantization for BoTNet modules (Case 2)."
    )
    parser.add_argument(
        "--skip-ca-quant",
        action="store_true",
        help="Skip INT8 quantization for CoordAtt modules (Case 3)."
    )
    parser.add_argument(
        "--skip-odconv-quant",
        action="store_true",
        help="Skip INT8 quantization for ODConv modules (Case 4)."
    )
    # ---------------------------------------------

    args = parser.parse_args()

    quantize_botnet_flag = not args.skip_botnet_quant
    quantize_coordatt_flag = not args.skip_ca_quant
    quantize_odconv_flag = not args.skip_odconv_quant

    results = train_ptq(
        model_cfg=args.model_cfg,
        data_cfg=args.data_cfg,
        weights=args.weights,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        backend=args.backend,
        save_dir=Path(args.save_dir) if args.save_dir else None,
        run_name=args.run_name,
        convert_to_int8=not args.no_convert,
        use_fx=not args.no_fx,
        num_calibration_batches=args.num_calibration_batches,
        calibration_split=args.calibration_split,
        evaluate=args.evaluate,
        quantize_botnet=quantize_botnet_flag,
        quantize_coordatt=quantize_coordatt_flag,
    )

    print("\n" + "=" * 80)
    print("PTQ Complete!")
    print("=" * 80)
    if results["int8_path"]:
        print(f"INT8 model saved to: {results['int8_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

