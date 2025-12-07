#!/usr/bin/env python3
"""
Convert QAT model to INT8 while preserving statistics from FakeQuantize modules.
This addresses the statistics loss issue for Conv2d inside fused wrappers.
"""

import sys
from pathlib import Path
sys.path.insert(0, 'obc-yolov8/ultralytics10.24')

import torch
from torch.ao.quantization import FakeQuantize, convert
from ultralytics import YOLO
from ultralytics.nn.tasks import ensure_module_bookkeeping, _create_safe_qconfig
from ultralytics.utils import LOGGER

def extract_qat_statistics(model):
    """
    Extract statistics from FakeQuantize modules BEFORE convert() removes them.
    
    Returns:
        dict: Mapping of module paths to (min_val, max_val) tuples
    """
    statistics = {}
    
    LOGGER.info("Extracting statistics from QAT model...")
    
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantize):
            try:
                observer = getattr(module, 'activation_post_process', None)
                if observer:
                    min_val = getattr(observer, 'min_val', None)
                    max_val = getattr(observer, 'max_val', None)
                    
                    if min_val is not None and max_val is not None:
                        # Clone to preserve values
                        if isinstance(min_val, torch.Tensor):
                            min_val = min_val.clone()
                        if isinstance(max_val, torch.Tensor):
                            max_val = max_val.clone()
                        
                        statistics[name] = (min_val, max_val)
            except Exception as e:
                LOGGER.debug(f"  Could not extract statistics from {name}: {e}")
    
    # Also extract from Conv2d modules that have activation_post_process
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                activation_fq = getattr(module, 'activation_post_process', None)
                if isinstance(activation_fq, FakeQuantize):
                    observer = getattr(activation_fq, 'activation_post_process', None)
                    if observer:
                        min_val = getattr(observer, 'min_val', None)
                        max_val = getattr(observer, 'max_val', None)
                        
                        if min_val is not None and max_val is not None:
                            if isinstance(min_val, torch.Tensor):
                                min_val = min_val.clone()
                            if isinstance(max_val, torch.Tensor):
                                max_val = max_val.clone()
                            
                            # Store with .activation_post_process suffix to match later
                            statistics[f"{name}.activation_post_process"] = (min_val, max_val)
            except Exception as e:
                LOGGER.debug(f"  Could not extract statistics from {name}.activation_post_process: {e}")
    
    LOGGER.info(f"  Extracted statistics from {len(statistics)} FakeQuantize modules")
    return statistics

def convert_qat_with_preserved_stats(qat_checkpoint_path, output_path, imgsz=640, target_backend=None):
    """
    Convert QAT model to INT8 while preserving statistics.
    
    Args:
        qat_checkpoint_path: Path to QAT checkpoint
        output_path: Path to save INT8 checkpoint
        imgsz: Image size
        target_backend: Target backend for conversion ('fbgemm' or 'qnnpack'). 
                       If None, uses the backend from the checkpoint.
    """
    qat_path = Path(qat_checkpoint_path)
    if not qat_path.exists():
        raise FileNotFoundError(f"QAT checkpoint not found: {qat_checkpoint_path}")
    
    LOGGER.info("=" * 80)
    LOGGER.info(f"Converting QAT to INT8 with Preserved Statistics")
    LOGGER.info("=" * 80)
    
    # Load QAT model
    checkpoint = torch.load(qat_path, map_location='cpu', weights_only=False)
    source_backend = checkpoint.get('backend', 'qnnpack')
    
    # Use target_backend if provided, otherwise use source backend
    backend = target_backend if target_backend is not None else source_backend
    
    if source_backend != backend:
        LOGGER.info(f"⚠️  Switching backend: {source_backend} (training) -> {backend} (conversion)")
        LOGGER.info(f"   Statistics are backend-agnostic, but qconfig will use {backend}")
    
    if backend in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = backend
        LOGGER.info(f"✓ Set quantization backend to {backend}")
    else:
        LOGGER.warning(f"Backend '{backend}' not supported, using default: {torch.backends.quantized.engine}")
    
    model_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml'
    yolo = YOLO(model_cfg)
    model_obj = checkpoint['model']
    
    if isinstance(model_obj, dict):
        example_input = torch.randn(1, 3, imgsz, imgsz)
        yolo.model = yolo.model.prepare_for_qat(
            backend=backend,
            example_input=example_input,
        )
        yolo.model.load_state_dict(model_obj, strict=False)
    else:
        yolo.model = model_obj
    
    yolo.model.eval()
    yolo.model = yolo.model.float()
    
    # CRITICAL: Extract statistics BEFORE convert() removes FakeQuantize
    LOGGER.info("\nStep 1: Extracting statistics from QAT model...")
    qat_statistics = extract_qat_statistics(yolo.model)
    
    # CRITICAL: Manually convert Conv2d inside fused wrappers BEFORE calling convert_to_quantized()
    # This way we can use preserved statistics
    LOGGER.info("\nStep 2: Manually converting Conv2d modules inside fused wrappers with preserved statistics...")
    
    from ultralytics.nn.modules.conv import Conv, Conv2, DWConv
    from torch.ao.nn.quantized.modules.conv import Conv2d as QuantizedConv2d
    from torch.ao.nn.qat.modules.conv import Conv2d as QATConv2d
    from torch.ao.quantization import prepare_qat
    
    manually_converted = 0
    stats_used = 0
    
    # Find Conv2d modules inside fused wrappers BEFORE convert_to_quantized() runs
    # These modules don't have FakeQuantize attached, so convert() won't handle them
    for name, module in list(yolo.model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            parent_path = '.'.join(name.split('.')[:-1])
            if parent_path:
                try:
                    parent = dict(yolo.model.named_modules()).get(parent_path)
                    if parent is not None and isinstance(parent, (Conv, Conv2, DWConv)):
                        if not hasattr(parent, 'bn'):  # Fused
                            # CRITICAL: Convert BEFORE convert_to_quantized() runs
                            # These modules won't be converted by convert() because they lack FakeQuantize
                                # Need to convert - try to use preserved statistics
                                LOGGER.info(f"  Converting {name} with preserved statistics...")
                                
                                # Look for statistics for this module
                                # Statistics might be stored under parent path
                                activation_stats = None
                                
                                # Try different possible paths to find statistics
                                # Statistics are stored at paths like "model.0.conv.activation_post_process"
                                # The exact path is: {name}.activation_post_process
                                activation_stats = None
                                
                                # Direct lookup: statistics are stored at {name}.activation_post_process
                                stat_path = f"{name}.activation_post_process"
                                if stat_path in qat_statistics:
                                    activation_stats = qat_statistics[stat_path]
                                    LOGGER.info(f"    Found statistics for {stat_path}")
                                    stats_used += 1
                                else:
                                    # Try alternative paths
                                    possible_paths = [
                                        name,  # Full path like "model.0.conv"
                                        f"{parent_path}.conv.activation_post_process",  # Parent path + conv
                                    ]
                                    
                                    for stat_path in possible_paths:
                                        if stat_path in qat_statistics:
                                            activation_stats = qat_statistics[stat_path]
                                            LOGGER.info(f"    Found statistics for {stat_path}")
                                            stats_used += 1
                                            break
                                    
                                    # Also try to find by matching the Conv2d name
                                    if activation_stats is None:
                                        conv_name = name.split('.')[-1]  # e.g., "conv"
                                        for stat_name, stats in qat_statistics.items():
                                            # Check if this stat path matches our Conv2d
                                            if stat_name.endswith(f".{conv_name}.activation_post_process") or \
                                               (parent_path and stat_name.startswith(parent_path) and f".{conv_name}.activation_post_process" in stat_name):
                                                activation_stats = stats
                                                LOGGER.info(f"    Found statistics for {stat_name} (matched to {name})")
                                                stats_used += 1
                                                break
                                
                                if activation_stats is None:
                                    LOGGER.warning(f"    No preserved statistics found for {name}, using dummy forward pass")
                                
                                # Create QATConv2d
                                qconfig = _create_safe_qconfig(backend)
                                qat_conv = QATConv2d(
                                    module.in_channels,
                                    module.out_channels,
                                    module.kernel_size,
                                    stride=module.stride,
                                    padding=module.padding,
                                    dilation=module.dilation,
                                    groups=module.groups,
                                    bias=module.bias is not None,
                                    padding_mode=module.padding_mode,
                                    qconfig=qconfig
                                )
                                
                                qat_conv.weight = torch.nn.Parameter(module.weight.data.clone())
                                if module.bias is not None:
                                    qat_conv.bias = torch.nn.Parameter(module.bias.data.clone())
                                
                                # Prepare QAT
                                qat_conv.train()
                                prepare_qat(qat_conv, inplace=True)
                                
                                # CRITICAL: Apply preserved statistics if available
                                if activation_stats is not None:
                                    min_val, max_val = activation_stats
                                    # Get the observer from the FakeQuantize
                                    activation_fq = getattr(qat_conv, 'activation_post_process', None)
                                    if activation_fq:
                                        # The observer is inside the FakeQuantize
                                        obs = getattr(activation_fq, 'activation_post_process', None)
                                        if obs and hasattr(obs, 'min_val') and hasattr(obs, 'max_val'):
                                            try:
                                                # Clone tensors to avoid sharing memory
                                                if isinstance(min_val, torch.Tensor):
                                                    # Ensure same device and dtype
                                                    if obs.min_val is not None:
                                                        target_device = obs.min_val.device if hasattr(obs.min_val, 'device') else 'cpu'
                                                        target_dtype = obs.min_val.dtype if hasattr(obs.min_val, 'dtype') else min_val.dtype
                                                        obs.min_val = min_val.clone().to(device=target_device, dtype=target_dtype)
                                                    else:
                                                        obs.min_val = min_val.clone()
                                                else:
                                                    # Scalar value
                                                    if obs.min_val is not None and hasattr(obs.min_val, 'fill_'):
                                                        obs.min_val.fill_(float(min_val))
                                                    else:
                                                        obs.min_val = torch.tensor(float(min_val))
                                                
                                                if isinstance(max_val, torch.Tensor):
                                                    if obs.max_val is not None:
                                                        target_device = obs.max_val.device if hasattr(obs.max_val, 'device') else 'cpu'
                                                        target_dtype = obs.max_val.dtype if hasattr(obs.max_val, 'dtype') else max_val.dtype
                                                        obs.max_val = max_val.clone().to(device=target_device, dtype=target_dtype)
                                                    else:
                                                        obs.max_val = max_val.clone()
                                                else:
                                                    if obs.max_val is not None and hasattr(obs.max_val, 'fill_'):
                                                        obs.max_val.fill_(float(max_val))
                                                    else:
                                                        obs.max_val = torch.tensor(float(max_val))
                                                
                                                LOGGER.info(f"    ✓ Applied preserved statistics to {name}")
                                            except Exception as e:
                                                LOGGER.warning(f"    Could not apply statistics: {e}, using dummy forward")
                                                import traceback
                                                LOGGER.debug(traceback.format_exc())
                                                activation_stats = None
                                
                                # If no preserved stats, use dummy forward
                                if activation_stats is None:
                                    dummy_input = torch.randn(1, module.in_channels, 3, 3)
                                    with torch.no_grad():
                                        _ = qat_conv(dummy_input)
                                
                                # Convert to INT8
                                qat_conv.eval()
                                quantized_conv = QuantizedConv2d.from_float(qat_conv)
                                
                                # Replace in parent
                                child_name = name.split('.')[-1]
                                setattr(parent, child_name, quantized_conv)
                                manually_converted += 1
                                
                except Exception as e:
                    LOGGER.debug(f"  Error processing {name}: {e}")
    
    LOGGER.info(f"\n  Manually converted {manually_converted} Conv2d modules")
    LOGGER.info(f"  Used preserved statistics for {stats_used} modules")
    
    # Now convert the rest of the model (this will handle modules with FakeQuantize)
    LOGGER.info("\nStep 3: Converting remaining QAT modules to INT8...")
    converted = yolo.model.convert_to_quantized()
    if converted is not None:
        yolo.model = converted
    
    ensure_module_bookkeeping(yolo.model, recursive=True)
    
    # Save
    int8_checkpoint = {
        'model': yolo.model,
        'backend': backend,
        'source_qat': str(qat_path),
        'preserved_stats': stats_used > 0,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(int8_checkpoint, output_path)
    LOGGER.info(f"\n✓ Saved INT8 checkpoint with preserved statistics to {output_path}")
    
    return str(output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--qat', type=str, required=True, help='QAT checkpoint path')
    parser.add_argument('--output', type=str, required=True, help='Output INT8 checkpoint path')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--backend', type=str, default=None, choices=['fbgemm', 'qnnpack'], 
                       help='Target backend for conversion (default: use backend from checkpoint)')
    
    args = parser.parse_args()
    convert_qat_with_preserved_stats(args.qat, args.output, args.imgsz, target_backend=args.backend)

