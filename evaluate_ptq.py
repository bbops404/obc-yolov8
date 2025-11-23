"""Evaluate a PTQ INT8 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.backends import quantized as torch_quantized_backends

# Ensure the local ultralytics package is importable
REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
if not ULTRALYTICS_PATH.exists():
    ULTRALYTICS_PATH = REPO_ROOT / "ultralytics10.24"
import sys

if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.nn import tasks as ultralytics_tasks

ensure_module_bookkeeping = getattr(
    ultralytics_tasks,
    "ensure_module_bookkeeping",
    lambda *args, **kwargs: None,
)

DEFAULT_DATA_CFG = ULTRALYTICS_PATH / "ultralytics" / "cfg" / "datasets" / "combined_china_motorbike.yaml"


def analyze_quantization_status(model, logger=None):
    """Analyze which modules are quantized vs FP32 and calculate sizes.
    
    Args:
        model: The model to analyze
        logger: Optional logger instance (defaults to LOGGER)
    
    Returns:
        dict with quantization statistics
    """
    if logger is None:
        logger = LOGGER
    
    import torch.nn as nn
    from ultralytics.nn.ODConv import ODConv
    from ultralytics.nn.BoTNet import BoTNet
    from ultralytics.nn.CA_Attention import CoordAtt
    
    quantized_layers = []
    fp32_layers = []
    total_params = 0
    quantized_params = 0
    fp32_params = 0
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_path = type(module).__module__
        
        # Check if quantized - quantized modules use _packed_params
        is_quantized = False
        quantized_weight_params = 0
        
        if module_type == 'Conv2d' and 'quantized' in module_path.lower():
            is_quantized = True
            # Get weight size from packed params
            if hasattr(module, '_packed_params'):
                try:
                    packed_params = module._packed_params
                    if hasattr(packed_params, 'weight'):
                        weight = packed_params.weight()
                        quantized_weight_params = weight.numel()
                except:
                    pass
        elif hasattr(module, 'conv') and isinstance(module.conv, nn.Module):
            conv_module = module.conv
            conv_path = type(conv_module).__module__
            if 'quantized' in conv_path.lower() or hasattr(conv_module, '_packed_params'):
                is_quantized = True
                if hasattr(conv_module, '_packed_params'):
                    try:
                        packed_params = conv_module._packed_params
                        if hasattr(packed_params, 'weight'):
                            weight = packed_params.weight()
                            quantized_weight_params = weight.numel()
                    except:
                        pass
        elif module_type == 'Linear' and 'quantized' in module_path.lower():
            is_quantized = True
            if hasattr(module, '_packed_params'):
                try:
                    packed_params = module._packed_params
                    if hasattr(packed_params, 'weight'):
                        weight = packed_params.weight()
                        quantized_weight_params = weight.numel()
                except:
                    pass
        
        # Count parameters
        module_params = sum(p.numel() for p in module.parameters())
        
        if is_quantized:
            # Use quantized weight count if available
            if quantized_weight_params > 0:
                param_count = quantized_weight_params
            else:
                # Fallback: estimate from module (may not be accurate for quantized)
                param_count = module_params
            quantized_layers.append((name, module_type, param_count))
            quantized_params += param_count
            # Don't add to total_params here - quantized modules may not have regular params
        else:
            # Check if it's a special FP32 module
            is_special_fp32 = isinstance(module, (ODConv, BoTNet, CoordAtt))
            fp32_layers.append((name, module_type, module_params, is_special_fp32))
            fp32_params += module_params
            total_params += module_params
    
    # Calculate actual model size from parameters
    # Note: Quantized modules use packed_params, so regular parameters() may not reflect true size
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    # For quantized modules, we need to check packed_params
    # Quantized Conv2d/Linear use _packed_params which stores INT8 weights
    quantized_weight_bytes = 0
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_path = type(module).__module__
        
        is_quantized = False
        if module_type == 'Conv2d' and 'quantized' in module_path.lower():
            is_quantized = True
        elif hasattr(module, 'conv') and isinstance(module.conv, nn.Module):
            conv_module = module.conv
            if hasattr(conv_module, '_packed_params'):
                is_quantized = True
        elif module_type == 'Linear' and 'quantized' in module_path.lower():
            is_quantized = True
        
        if is_quantized and hasattr(module, '_packed_params'):
            # Get weight size from packed params (INT8)
            packed_params = module._packed_params
            if hasattr(packed_params, 'weight'):
                weight = packed_params.weight()
                quantized_weight_bytes += weight.numel() * 1  # INT8 = 1 byte
    
    # Calculate sizes
    # Actual size: what's currently in memory (may include FP32 buffers for quantized modules)
    # Quantized size estimate: INT8 weights + scales/zero_points (FP32) + FP32 layers
    quantized_weight_mb = quantized_weight_bytes / (1024 * 1024)
    fp32_weight_mb = (fp32_params * 4) / (1024 * 1024)  # FP32 = 4 bytes
    # Add overhead for scales/zero_points (roughly 1 float per channel for per-channel quantization)
    estimated_quantized_mb = quantized_weight_mb + fp32_weight_mb
    
    return {
        'quantized_layers': quantized_layers,
        'fp32_layers': fp32_layers,
        'total_params': total_params,
        'quantized_params': quantized_params,
        'fp32_params': fp32_params,
        'model_size_mb': model_size_mb,
        'quantized_weight_mb': quantized_weight_mb,
        'fp32_weight_mb': fp32_weight_mb,
        'estimated_quantized_mb': estimated_quantized_mb,
    }


def print_quantization_summary(stats, logger=None):
    """Print a summary of quantization status."""
    if logger is None:
        logger = LOGGER
    
    logger.info("\n" + "=" * 80)
    logger.info("QUANTIZATION STATUS")
    logger.info("=" * 80)
    
    # Overall statistics
    quantized = stats['quantized_params']
    fp32 = stats['fp32_params']
    effective_total = quantized + fp32  # Effective total
    
    logger.info(f"\nParameter Count (weights only):")
    if effective_total > 0:
        logger.info(f"  Quantized (INT8):    {quantized:,} ({100*quantized/effective_total:.2f}%)")
        logger.info(f"  FP32:                {fp32:,} ({100*fp32/effective_total:.2f}%)")
        logger.info(f"  Total:               {effective_total:,}")
    else:
        logger.info(f"  Quantized (INT8):    {quantized:,}")
        logger.info(f"  FP32:                {fp32:,}")
    logger.info(f"  Note: Quantized weights are INT8 (1 byte), FP32 weights are 4 bytes")
    
    logger.info(f"\nModel Size:")
    logger.info(f"  Actual size (in memory):     {stats['model_size_mb']:.2f} MB")
    logger.info(f"  Quantized weights (INT8):     {stats['quantized_weight_mb']:.2f} MB")
    logger.info(f"  FP32 weights:                 {stats['fp32_weight_mb']:.2f} MB")
    logger.info(f"  Estimated total (INT8+FP32):  {stats['estimated_quantized_mb']:.2f} MB")
    logger.info(f"  Note: Actual size includes PyTorch overhead and buffers")
    
    # Quantized layers
    logger.info(f"\nQuantized Layers: {len(stats['quantized_layers'])}")
    if stats['quantized_layers']:
        conv_layers = [l for l in stats['quantized_layers'] if 'Conv2d' in l[1]]
        linear_layers = [l for l in stats['quantized_layers'] if 'Linear' in l[1]]
        logger.info(f"  - Conv2d: {len(conv_layers)}")
        logger.info(f"  - Linear: {len(linear_layers)}")
    
    # FP32 layers (focus on special modules)
    special_fp32 = [l for l in stats['fp32_layers'] if l[3]]  # is_special_fp32 flag
    if special_fp32:
        logger.info(f"\nSpecial FP32 Modules (kept in FP32):")
        from collections import defaultdict
        module_counts = defaultdict(int)
        for name, mod_type, params, _ in special_fp32:
            module_counts[mod_type] += 1
        for mod_type, count in module_counts.items():
            logger.info(f"  - {mod_type}: {count}")
    
    logger.info("=" * 80)


def evaluate_int8_model(
    int8_weights: str | Path,
    data_cfg: str | Path = DEFAULT_DATA_CFG,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    backend: str = "qnnpack",
):
    """Evaluate an INT8 quantized model.
    
    Args:
        int8_weights: Path to INT8 model weights (.pt file)
        data_cfg: Path to dataset YAML configuration
        imgsz: Image size for evaluation
        batch: Batch size for evaluation
        device: Device to use (must be 'cpu' for qnnpack backend)
        backend: Quantization backend
    """
    int8_path = Path(int8_weights)
    if not int8_path.exists():
        raise FileNotFoundError(f"INT8 model not found: {int8_path}")
    
    LOGGER.info("=" * 80)
    LOGGER.info("Evaluating INT8 PTQ Model")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Model: {int8_path}")
    LOGGER.info(f"Backend: {backend}")
    LOGGER.info(f"Device: {device}")
    LOGGER.info("=" * 80)
    
    # Set quantization backend engine BEFORE loading model
    if backend in torch_quantized_backends.supported_engines:
        torch_quantized_backends.engine = backend
        LOGGER.info(f"Set quantization backend engine to {backend}")
    else:
        LOGGER.warning(f"Backend '{backend}' not supported, using default: {torch_quantized_backends.engine}")
    
    # Load INT8 model first to check its backend
    LOGGER.info(f"\nLoading INT8 model from {int8_path}...")
    checkpoint = torch.load(int8_path, map_location='cpu', weights_only=False)
    
    # Check if checkpoint has a backend specified - use it if available
    checkpoint_backend = checkpoint.get('backend', None)
    if checkpoint_backend:
        LOGGER.info(f"Checkpoint specifies backend: {checkpoint_backend}")
        # Use checkpoint's backend if it's supported, otherwise use provided backend
        if checkpoint_backend in torch_quantized_backends.supported_engines:
            backend = checkpoint_backend
            LOGGER.info(f"Using checkpoint backend: {backend}")
        else:
            LOGGER.warning(f"Checkpoint backend '{checkpoint_backend}' not supported, using provided: {backend}")
    
    # Re-set quantization backend engine with the correct backend
    # CRITICAL: This must be set before any quantized operations are attempted
    if backend in torch_quantized_backends.supported_engines:
        torch_quantized_backends.engine = backend
        LOGGER.info(f"Set quantization backend engine to {backend}")
        # Verify it was set correctly
        actual_engine = torch_quantized_backends.engine
        if actual_engine != backend:
            LOGGER.warning(f"Backend setting may have failed. Expected: {backend}, Actual: {actual_engine}")
    else:
        LOGGER.warning(f"Backend '{backend}' not supported, using default: {torch.backends.quantized.engine}")
    
    # Determine how to load the model
    model_yaml_path = None
    if 'model' in checkpoint:
        # Full model object saved
        LOGGER.info("Loading full model object...")
        int8_model = checkpoint['model']
        model_yaml = checkpoint.get('yaml', None)
        
        # Handle YAML - could be a dict or a path
        if isinstance(model_yaml, dict):
            # Extract yaml_file path from dict if available
            model_yaml_path = model_yaml.get('yaml_file', None)
            if model_yaml_path:
                # Resolve relative path
                if not Path(model_yaml_path).is_absolute():
                    model_yaml_path = str(ULTRALYTICS_PATH / model_yaml_path)
        elif isinstance(model_yaml, (str, Path)):
            model_yaml_path = str(model_yaml)
    elif 'model_state_dict' in checkpoint:
        # Only state_dict saved - need to reconstruct model
        LOGGER.info("Reconstructing model from state_dict...")
        model_yaml = checkpoint.get('yaml', None)
        if model_yaml is None:
            raise ValueError("Model YAML not found in checkpoint. Cannot reconstruct model.")
        
        # Handle YAML - could be a dict or a path
        if isinstance(model_yaml, dict):
            model_yaml_path = model_yaml.get('yaml_file', None)
            if model_yaml_path:
                if not Path(model_yaml_path).is_absolute():
                    model_yaml_path = str(ULTRALYTICS_PATH / model_yaml_path)
        elif isinstance(model_yaml, (str, Path)):
            model_yaml_path = str(model_yaml)
        
        if model_yaml_path is None:
            raise ValueError("Could not extract YAML file path from checkpoint.")
        
        # Reconstruct model from YAML
        model = YOLO(str(model_yaml_path))
        detection_model = getattr(model, 'model', None)
        
        if detection_model is None:
            raise RuntimeError("Failed to create DetectionModel from YAML")
        
        # Load state_dict
        detection_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        int8_model = detection_model
    else:
        raise ValueError("Checkpoint must contain either 'model' or 'model_state_dict'")
    
    # Ensure bookkeeping for quantized modules
    LOGGER.info("Ensuring module bookkeeping...")
    ensure_module_bookkeeping(int8_model, recursive=True)
    
    # Additional fix for quantized modules - ensure all hook attributes exist
    # Quantized modules sometimes miss these attributes which causes AttributeError
    LOGGER.info("Fixing quantized module bookkeeping attributes...")
    import torch.nn as nn
    from collections import OrderedDict
    
    hook_attrs = [
        '_backward_hooks',
        '_backward_pre_hooks', 
        '_forward_hooks',
        '_forward_pre_hooks',
        '_state_dict_hooks',
        '_load_state_dict_pre_hooks',
        '_state_dict_pre_hooks',
        '_load_state_dict_hooks',
    ]
    
    fixed_count = 0
    for name, module in int8_model.named_modules():
        if isinstance(module, nn.Module):
            for attr_name in hook_attrs:
                try:
                    # Check if attribute exists and is accessible
                    if not hasattr(module, attr_name):
                        setattr(module, attr_name, OrderedDict())
                        fixed_count += 1
                    else:
                        # Ensure it's an OrderedDict (not None or wrong type)
                        current = getattr(module, attr_name, None)
                        if not isinstance(current, (OrderedDict, dict)):
                            setattr(module, attr_name, OrderedDict())
                            fixed_count += 1
                except (AttributeError, RuntimeError):
                    # Some modules may not allow setting these attributes
                    pass
    
    if fixed_count > 0:
        LOGGER.info(f"Fixed {fixed_count} missing/invalid bookkeeping attributes in quantized modules")
    
    # Analyze quantization status
    LOGGER.info("\nAnalyzing quantization status...")
    quant_stats = analyze_quantization_status(int8_model, logger=LOGGER)
    print_quantization_summary(quant_stats, logger=LOGGER)
    
    # Create YOLO wrapper - use the model directly if we have it, otherwise reconstruct
    if 'model' in checkpoint:
        # We already have the model, just need to wrap it
        # Create a minimal YOLO wrapper
        model = YOLO(model=model_yaml_path) if model_yaml_path else YOLO()
        model.model = int8_model
    else:
        # Already created model above
        model.model = int8_model
    
    # Move to device
    if backend in ['qnnpack']:
        device = "cpu"  # qnnpack requires CPU
        LOGGER.info("Using CPU (required for qnnpack backend)")
    
    model.to(torch.device(device))
    
    # Set eval mode
    try:
        int8_model.eval()
        LOGGER.info("Model set to eval mode")
    except (AttributeError, RuntimeError) as e:
        LOGGER.info(f"Model already in eval mode or quantized structure: {type(e).__name__}")
    
    # Run evaluation
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Running evaluation...")
    LOGGER.info("=" * 80)
    
    eval_results = model.val(
        data=str(data_cfg),
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=False,
        save=False,
        verbose=True
    )
    
    # Print results
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Evaluation Results")
    LOGGER.info("=" * 80)
    
    if eval_results:
        # Extract metrics
        if isinstance(eval_results, dict):
            map50 = eval_results.get('metrics/mAP50(B)', eval_results.get('map50', None))
            map = eval_results.get('metrics/mAP50-95(B)', eval_results.get('map', None))
            precision = eval_results.get('metrics/precision(B)', eval_results.get('precision', None))
            recall = eval_results.get('metrics/recall(B)', eval_results.get('recall', None))
        else:
            # DetMetrics object
            map50 = getattr(eval_results, 'map50', None)
            map = getattr(eval_results, 'map', None)
            precision = getattr(eval_results, 'precision', None)
            recall = getattr(eval_results, 'recall', None)
            
            # Try accessing via metrics attribute
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
            LOGGER.info(f"  mAP@0.5:0.95:  {map:.4f} ({map*100:.2f}%)")
        if precision is not None:
            LOGGER.info(f"  Precision:    {precision:.4f} ({precision*100:.2f}%)")
        if recall is not None:
            LOGGER.info(f"  Recall:       {recall:.4f} ({recall*100:.2f}%)")
    else:
        LOGGER.warning("Evaluation completed but no metrics returned")
    
    LOGGER.info("=" * 80)
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate INT8 PTQ model")
    parser.add_argument(
        "--int8-weights",
        type=str,
        required=True,
        help="Path to INT8 model weights (.pt file)",
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
        help="Image size for evaluation",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (must be 'cpu' for qnnpack backend)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="qnnpack",
        choices=["qnnpack"],
        help="Quantization backend",
    )
    
    args = parser.parse_args()
    
    evaluate_int8_model(
        int8_weights=args.int8_weights,
        data_cfg=args.data_cfg,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()

