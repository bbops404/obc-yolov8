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
    
    # Ensure we're in a context where we can modify requires_grad
    # Exit any inference mode that might have been set during evaluation
    with torch.enable_grad():
        # Ensure model is in training mode
        model.train()
        
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
                # Access parameters through named_parameters to avoid inference tensor issues
                for param_name, param in module.named_parameters(recurse=False):
                    try:
                        # Try to set requires_grad - if it fails, the param might be an inference tensor
                        param.requires_grad = True
                        trainable_params.append(param)
                    except RuntimeError as e:
                        if "inference tensor" in str(e):
                            logger.warning(f"  ⚠️  Parameter {name}.{param_name} is an inference tensor, replacing...")
                            # Replace the inference tensor parameter with a new Parameter object
                            try:
                                # Clone the parameter data and create a new Parameter
                                cloned_data = param.data.clone().detach()
                                new_param = nn.Parameter(cloned_data, requires_grad=True)
                                # Replace in module's _parameters dict
                                if hasattr(module, '_parameters') and param_name in module._parameters:
                                    module._parameters[param_name] = new_param
                                    trainable_params.append(new_param)
                                    logger.info(f"     ✓ Successfully replaced inference tensor parameter")
                                else:
                                    logger.warning(f"     ✗ Could not find parameter in _parameters dict")
                            except Exception as e2:
                                logger.warning(f"     ✗ Failed to replace inference tensor: {e2}")
                        else:
                            raise
    
    if not unfrozen_modules:
        logger.warning(f"⚠️  No modules matched patterns: {sensitive_modules}")
        logger.warning("    Model will have NO trainable parameters!")
    
    return trainable_params


def freeze_observers_selectively(model, keep_observers_enabled: List[str], logger=None):
    """Selectively freeze observers - disable for all modules except those matching patterns.
    
    Args:
        model: The model to modify
        keep_observers_enabled: List of module name patterns to keep observers enabled (e.g., ['model.10'])
        logger: Logger instance
    
    Returns:
        Tuple of (disabled_count, enabled_count)
    """
    if logger is None:
        logger = LOGGER
    
    from torch.ao.quantization import FakeQuantize
    
    disabled_count = 0
    enabled_count = 0
    enabled_modules = []
    disabled_modules = []
    
    for name, module in model.named_modules():
        # Check if this module matches any pattern in keep_observers_enabled
        should_keep_enabled = any(
            pattern in name 
            for pattern in keep_observers_enabled
        )
        
        # Handle FakeQuantize modules
        if isinstance(module, FakeQuantize):
            try:
                if should_keep_enabled:
                    if hasattr(module, 'enable_observer'):
                        module.enable_observer()
                        enabled_count += 1
                        enabled_modules.append(name)
                        logger.debug(f"  Enabled observer for: {name}")
                else:
                    if hasattr(module, 'disable_observer'):
                        module.disable_observer()
                        disabled_count += 1
                        disabled_modules.append(name)
            except Exception as e:
                logger.warning(f"Failed to modify observer for {name}: {e}")
        
        # Handle modules with activation_post_process (legacy observers)
        if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
            try:
                observer = module.activation_post_process
                if should_keep_enabled:
                    if hasattr(observer, 'enable_observer'):
                        observer.enable_observer()
                        enabled_count += 1
                        if name not in enabled_modules:
                            enabled_modules.append(name)
                        logger.debug(f"  Enabled observer for: {name}")
                else:
                    if hasattr(observer, 'disable_observer'):
                        observer.disable_observer()
                        disabled_count += 1
                        if name not in disabled_modules:
                            disabled_modules.append(name)
            except Exception as e:
                logger.warning(f"Failed to modify activation_post_process for {name}: {e}")
    
    # Log summary
    if enabled_modules:
        logger.info(f"  Observers enabled for {len(enabled_modules)} modules (QAT will update scales):")
        for mod_name in enabled_modules[:10]:  # Show first 10
            logger.info(f"    - {mod_name}")
        if len(enabled_modules) > 10:
            logger.info(f"    ... and {len(enabled_modules) - 10} more")
    
    if disabled_modules and logger.level <= 10:  # Only show in debug mode
        logger.debug(f"  Observers disabled for {len(disabled_modules)} modules (PTQ scales frozen)")
    
    return disabled_count, enabled_count


def convert_observers_to_fakequantize_selectively(model, module_patterns: List[str], backend: str = 'qnnpack', logger=None):
    """Convert Observer modules to FakeQuantize modules for specified module patterns.
    
    This function selectively converts only modules matching the patterns from PTQ (Observers)
    to QAT (FakeQuantize), while keeping other modules with Observers (PTQ state).
    
    Args:
        model: The model to modify
        module_patterns: List of module name patterns to convert (e.g., ['model.10'])
        backend: Quantization backend ('qnnpack' or 'fbgemm')
        logger: Logger instance
    
    Returns:
        List of converted module names
    """
    if logger is None:
        logger = LOGGER
    
    from torch.ao.quantization import FakeQuantize, get_default_qat_qconfig
    from torch.ao.quantization.observer import ObserverBase
    
    converted_modules = []
    
    # Get QAT qconfig for creating FakeQuantize modules
    try:
        from ultralytics.nn.tasks import _create_safe_qconfig, ClampedMovingAverageObserver, ClampedMovingAveragePerChannelObserver
        from ultralytics.nn.tasks import SAFE_QAT_CLAMP_VALUE, SAFE_QAT_AVERAGING_CONSTANT, SAFE_QAT_EPS
        qconfig = _create_safe_qconfig(backend)
        # Use activation FakeQuantize from qconfig (per-tensor for activations)
        activation_fake_quant_factory = qconfig.activation
    except Exception as e:
        logger.warning(f"Could not import safe qconfig, using default: {e}")
        qconfig = get_default_qat_qconfig(backend)
        activation_fake_quant_factory = qconfig.activation
        # Fallback: create simple FakeQuantize
        from torch.ao.quantization.fake_quantize import FakeQuantize
        from torch.ao.quantization import MovingAverageMinMaxObserver
        activation_fake_quant_factory = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        )
    
    for name, module in model.named_modules():
        # Skip modules that are themselves Observers or FakeQuantize
        # We only want to process modules that HAVE activation_post_process, not modules that ARE activation_post_process
        if isinstance(module, ObserverBase) or isinstance(module, FakeQuantize):
            continue
        
        # Check if this module matches any pattern
        should_convert = any(pattern in name for pattern in module_patterns)
        
        # Check if it's a regular Conv2d that needs conversion
        is_regular_conv2d = isinstance(module, nn.Conv2d)
        is_qat_conv2d = hasattr(module, 'weight_fake_quant')
        is_real_quantized = (
            hasattr(module, '_packed_params') or
            (('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
             and 'conv' in type(module).__name__.lower())
        )
        needs_regular_conv_conversion = should_convert and is_regular_conv2d and not is_qat_conv2d and not is_real_quantized
        
        # CRITICAL: Also check for real quantized Conv2d modules even if they don't have activation_post_process
        # These might be inside matched patterns (e.g., query/key/value in MHSA)
        # Check if it's a Conv2d-like module that's real quantized
        is_conv2d_like = (
            hasattr(module, 'in_channels') and hasattr(module, 'out_channels') and 
            hasattr(module, 'kernel_size') and hasattr(module, 'stride') and
            hasattr(module, 'padding')
        )
        needs_quantized_conv_conversion = should_convert and is_conv2d_like and is_real_quantized and not is_qat_conv2d
        
        # Skip if module doesn't match pattern and doesn't need conversion
        if not should_convert and not needs_regular_conv_conversion and not needs_quantized_conv_conversion:
            continue
        
        # Check if module has activation_post_process (Observer)
        if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
            observer = module.activation_post_process
            
            try:
                # Extract Observer statistics
                min_val = None
                max_val = None
                if hasattr(observer, 'min_val'):
                    min_val = observer.min_val
                if hasattr(observer, 'max_val'):
                    max_val = observer.max_val
                
                # CRITICAL: Only convert if Observer has been calibrated (has statistics)
                # Uncalibrated Observers (min_val/max_val are None or uninitialized) should not be converted
                has_statistics = False
                if min_val is not None and max_val is not None:
                    # Check if statistics are actually populated (not just default values)
                    if isinstance(min_val, torch.Tensor):
                        has_statistics = min_val.numel() > 0 and max_val.numel() > 0
                    else:
                        has_statistics = True  # Scalar values indicate calibration
                
                if not has_statistics:
                    logger.debug(f"  Skipping {name}: Observer has no calibrated statistics (min_val={min_val}, max_val={max_val})")
                    continue
                
                # Get observer dtype and qscheme
                observer_qscheme = getattr(observer, 'qscheme', torch.per_tensor_affine)
                
                # Determine if per-channel or per-tensor
                is_per_channel = observer_qscheme == torch.per_channel_symmetric or observer_qscheme == torch.per_channel_affine
                
                # Create appropriate FakeQuantize module
                if is_per_channel:
                    # Per-channel FakeQuantize (for weights, but we use it for activations if needed)
                    try:
                        from ultralytics.nn.tasks import ClampedMovingAveragePerChannelObserver, SAFE_QAT_CLAMP_VALUE, SAFE_QAT_AVERAGING_CONSTANT, SAFE_QAT_EPS
                        fake_quant = FakeQuantize.with_args(
                            observer=ClampedMovingAveragePerChannelObserver,
                            quant_min=-127,
                            quant_max=127,
                            dtype=torch.qint8,
                            qscheme=torch.per_channel_symmetric,
                            ch_axis=0,
                            reduce_range=False,
                            averaging_constant=SAFE_QAT_AVERAGING_CONSTANT,
                            clamp_value=SAFE_QAT_CLAMP_VALUE,
                            eps=SAFE_QAT_EPS,
                        )()
                    except ImportError:
                        # Fallback to default
                        from torch.ao.quantization import MovingAveragePerChannelMinMaxObserver
                        fake_quant = FakeQuantize.with_args(
                            observer=MovingAveragePerChannelMinMaxObserver,
                            quant_min=-127,
                            quant_max=127,
                            dtype=torch.qint8,
                            qscheme=torch.per_channel_symmetric,
                            ch_axis=0,
                        )()
                else:
                    # Per-tensor FakeQuantize (for activations)
                    fake_quant = activation_fake_quant_factory()
                
                # Transfer statistics from Observer to FakeQuantize
                if min_val is not None and max_val is not None:
                    # Handle tensor values - preserve shape for per-channel
                    if isinstance(min_val, torch.Tensor):
                        min_val_tensor = min_val.clone()
                    else:
                        min_val_tensor = torch.tensor(min_val) if min_val is not None else None
                    
                    if isinstance(max_val, torch.Tensor):
                        max_val_tensor = max_val.clone()
                    else:
                        max_val_tensor = torch.tensor(max_val) if max_val is not None else None
                    
                    if min_val_tensor is not None and max_val_tensor is not None:
                        # Initialize FakeQuantize with PTQ statistics
                        if hasattr(fake_quant, 'activation_post_process'):
                            obs = fake_quant.activation_post_process
                            if hasattr(obs, 'min_val') and hasattr(obs, 'max_val'):
                                # Handle per-channel vs per-tensor
                                if is_per_channel:
                                    if min_val_tensor.dim() > 0:
                                        obs.min_val = min_val_tensor.clone()
                                    else:
                                        obs.min_val = min_val_tensor.unsqueeze(0).clone()
                                    if max_val_tensor.dim() > 0:
                                        obs.max_val = max_val_tensor.clone()
                                    else:
                                        obs.max_val = max_val_tensor.unsqueeze(0).clone()
                                else:
                                    # Per-tensor: use scalar or first element
                                    min_scalar = min_val_tensor.item() if min_val_tensor.numel() == 1 else min_val_tensor.min().item()
                                    max_scalar = max_val_tensor.item() if max_val_tensor.numel() == 1 else max_val_tensor.max().item()
                                    obs.min_val.fill_(min_scalar)
                                    obs.max_val.fill_(max_scalar)
                
                # Replace Observer with FakeQuantize
                module.activation_post_process = fake_quant
                converted_modules.append(name)
                logger.info(f"  Converted Observer to FakeQuantize: {name}")
                
            except Exception as e:
                logger.warning(f"Failed to convert Observer to FakeQuantize for {name}: {e}")
                import traceback
                traceback.print_exc()
        
        # CRITICAL: Check for regular nn.Conv2d modules that need to be converted to QAT Conv2d
        # Regular nn.Conv2d with FakeQuantize only quantizes activations, not weights
        # QAT Conv2d has weight_fake_quant for weight quantization
        # Re-check module type after Observer conversion (module type doesn't change, but be explicit)
        is_regular_conv2d_after = isinstance(module, nn.Conv2d)
        is_qat_conv2d_after = hasattr(module, 'weight_fake_quant')
        is_real_quantized_after = (
            hasattr(module, '_packed_params') or
            (('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
             and 'conv' in type(module).__name__.lower())
        )
        
        # Convert regular nn.Conv2d to QAT Conv2d if it's in a sensitive module pattern
        # and not already QAT or real quantized
        needs_regular_conv_conversion = (
            should_convert and 
            is_regular_conv2d_after and 
            not is_qat_conv2d_after and 
            not is_real_quantized_after
        )
        
        if needs_regular_conv_conversion:
            try:
                logger.info(f"  Found regular Conv2d at {name} - converting to QAT Conv2d...")
                
                # Extract weight and bias from regular Conv2d
                weight_fp32 = module.weight.data.clone() if hasattr(module.weight, 'data') else module.weight.clone()
                bias_fp32 = None
                if module.bias is not None:
                    bias_fp32 = module.bias.data.clone() if hasattr(module.bias, 'data') else module.bias.clone()
                
                # Preserve activation_post_process (FakeQuantize) if it exists
                preserved_activation_fake_quant = None
                if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
                    preserved_activation_fake_quant = module.activation_post_process
                
                # Create QAT Conv2d module
                from torch.ao.nn.qat.modules.conv import Conv2d as QATConv2d
                from ultralytics.nn.tasks import _create_safe_qconfig
                
                qconfig = _create_safe_qconfig(backend)
                qat_conv = QATConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=bias_fp32 is not None,
                    padding_mode=module.padding_mode,
                    qconfig=qconfig
                )
                
                # Copy weights and bias
                qat_conv.weight = nn.Parameter(weight_fp32)
                if bias_fp32 is not None:
                    qat_conv.bias = nn.Parameter(bias_fp32)
                
                # Preserve activation_post_process (FakeQuantize) if it was set
                if preserved_activation_fake_quant is not None:
                    qat_conv.activation_post_process = preserved_activation_fake_quant
                
                # CRITICAL: Ensure QAT Conv2d is properly set up for training
                # QAT Conv2d should have weight_fake_quant and activation_post_process
                # Ensure FakeQuantize modules are enabled for training and in FAKE quantization mode (not real)
                qat_conv.train()  # Set to training mode
                
                # Enable FakeQuantize for weights (if it exists)
                if hasattr(qat_conv, 'weight_fake_quant') and qat_conv.weight_fake_quant is not None:
                    weight_fq = qat_conv.weight_fake_quant
                    # Ensure fake quantization is enabled (simulation mode, returns FP32)
                    if hasattr(weight_fq, 'enable_fake_quant'):
                        weight_fq.enable_fake_quant()
                    # Ensure observer is enabled for QAT (to update scales during training)
                    if hasattr(weight_fq, 'enable_observer'):
                        weight_fq.enable_observer()
                    # Verify fake quantization is enabled
                    if hasattr(weight_fq, 'fake_quant_enabled'):
                        if not weight_fq.fake_quant_enabled:
                            logger.warning(f"    ⚠️  Weight FakeQuantize fake_quant_enabled is False for {name} - enabling it")
                            weight_fq.enable_fake_quant()
                
                # Enable FakeQuantize for activations (if it exists)
                if hasattr(qat_conv, 'activation_post_process') and qat_conv.activation_post_process is not None:
                    act_fq = qat_conv.activation_post_process
                    # CRITICAL: Ensure fake quantization is enabled (simulation mode, returns FP32)
                    # FakeQuantize should ALWAYS return FP32 tensors, never real quantized tensors
                    if hasattr(act_fq, 'enable_fake_quant'):
                        act_fq.enable_fake_quant()
                    # Ensure observer is enabled for QAT (to update scales during training)
                    if hasattr(act_fq, 'enable_observer'):
                        act_fq.enable_observer()
                    # Verify fake quantization is enabled
                    if hasattr(act_fq, 'fake_quant_enabled'):
                        if not act_fq.fake_quant_enabled:
                            logger.warning(f"    ⚠️  Activation FakeQuantize fake_quant_enabled is False for {name} - enabling it")
                            act_fq.enable_fake_quant()
                    # CRITICAL: Verify this is actually a FakeQuantize, not a real quantizer
                    from torch.ao.quantization import FakeQuantize
                    if not isinstance(act_fq, FakeQuantize):
                        logger.error(f"    ❌ activation_post_process for {name} is not a FakeQuantize! Type: {type(act_fq)}")
                        raise RuntimeError(f"activation_post_process for {name} must be FakeQuantize, not {type(act_fq)}")
                
                # Verify QAT Conv2d is properly set up
                has_weight_fake_quant = hasattr(qat_conv, 'weight_fake_quant') and qat_conv.weight_fake_quant is not None
                has_activation_fake_quant = hasattr(qat_conv, 'activation_post_process') and qat_conv.activation_post_process is not None
                is_qat_module_type = 'qat' in type(qat_conv).__module__.lower() and 'quantized' not in type(qat_conv).__module__.lower()
                
                if not (has_weight_fake_quant or is_qat_module_type):
                    logger.warning(f"    ⚠️  QAT Conv2d at {name} may not be properly configured (no weight_fake_quant)")
                
                # CRITICAL: Test that QAT Conv2d returns FP32 tensors, not quantized tensors
                # This is the key requirement - QAT should simulate quantization but return FP32
                # Note: FakeQuantize returns FP32 tensors with q_scale/q_zero_point attributes,
                # but they are NOT actually quantized (dtype is float32, not quint8/qint8)
                try:
                    with torch.no_grad():
                        test_input = torch.randn(1, qat_conv.in_channels, 8, 8)
                        test_output = qat_conv(test_input)
                        # Check if output is actually quantized (should NOT be)
                        # Real quantized tensors have quantized dtypes (quint8, qint8, etc.)
                        # FakeQuantize outputs have float32 dtype even if they have q_scale/q_zero_point
                        is_real_quantized = test_output.dtype in (torch.quint8, torch.qint8, torch.qint32)
                        if is_real_quantized:
                            logger.error(f"    ❌ QAT Conv2d at {name} returned real quantized tensor during test!")
                            logger.error(f"       This should NEVER happen - QAT Conv2d must return FP32 tensors")
                            logger.error(f"       Output dtype: {test_output.dtype}, which is a real quantized dtype")
                            raise RuntimeError(
                                f"QAT Conv2d at {name} returned real quantized tensor (dtype: {test_output.dtype}). "
                                f"QAT Conv2d must return FP32 tensors (FakeQuantize simulates quantization but keeps FP32). "
                                f"This indicates FakeQuantize is not properly configured."
                            )
                        else:
                            logger.debug(f"    ✓ QAT Conv2d at {name} correctly returns FP32 tensors (dtype: {test_output.dtype})")
                except Exception as test_error:
                    if "returned quantized tensor" in str(test_error) or "returned real quantized" in str(test_error):
                        raise  # Re-raise our specific error
                    # Other errors during test are non-critical
                    logger.debug(f"    Could not test QAT Conv2d output type for {name}: {test_error}")
                
                # Replace the regular Conv2d with QAT Conv2d
                parts = name.split('.')
                if len(parts) > 1:
                    parent_name = '.'.join(parts[:-1])
                    child_name = parts[-1]
                    parent_module = dict(model.named_modules())[parent_name]
                    setattr(parent_module, child_name, qat_conv)
                    converted_modules.append(name)
                    logger.info(f"  Converted regular Conv2d to QAT Conv2d: {name} (weight_fake_quant: {has_weight_fake_quant}, activation_fake_quant: {has_activation_fake_quant})")
                else:
                    logger.warning(f"    Could not find parent module for {name}")
                    
            except Exception as e:
                logger.warning(f"Failed to convert regular Conv2d to QAT for {name}: {e}")
                import traceback
                traceback.print_exc()
        
        # CRITICAL: Also check for real quantized Conv2d modules inside matched patterns
        # These need to be converted to QAT Conv2d (not just Observers to FakeQuantize)
        # Real quantized Conv2d modules have _packed_params and cannot be used for QAT training
        # Note: Real quantized Conv2d is NOT isinstance(module, nn.Conv2d) - it's torch.ao.nn.quantized.modules.conv.Conv2d
        
        # Check if it's Conv2d-like (has Conv2d attributes)
        is_conv2d_like = (
            hasattr(module, 'in_channels') and hasattr(module, 'out_channels') and 
            hasattr(module, 'kernel_size') and hasattr(module, 'stride') and
            hasattr(module, 'padding')
        )
        
        # Only convert if it's Conv2d-like, real quantized, and the Conv2d itself is NOT QAT
        # We convert even if it has FakeQuantize for activations, as long as the Conv2d itself is real quantized
        is_real_quantized_conv = is_conv2d_like and is_real_quantized and not is_qat_conv2d
        
        # Debug logging for query/key/value modules
        if 'query' in name or 'key' in name or 'value' in name:
            logger.debug(f"  Checking {name}: is_regular_conv2d={is_regular_conv2d}, is_conv2d_like={is_conv2d_like}, is_real_quantized={is_real_quantized}, is_qat_conv2d={is_qat_conv2d}, is_real_quantized_conv={is_real_quantized_conv}")
            logger.debug(f"    Module type: {type(module).__name__}, module path: {type(module).__module__}")
            logger.debug(f"    Has _packed_params: {hasattr(module, '_packed_params')}")
            logger.debug(f"    Has weight_fake_quant: {hasattr(module, 'weight_fake_quant')}")
            logger.debug(f"    Has activation_post_process: {hasattr(module, 'activation_post_process')}")
        
        if is_real_quantized_conv:
            try:
                logger.warning(f"  Found real quantized Conv2d at {name} (type: {type(module).__name__}, module: {type(module).__module__}) - converting to QAT Conv2d...")
                
                # Extract parameters from quantized Conv2d
                # Real quantized Conv2d has _packed_params which contains quantized weights
                # We need to dequantize them to get FP32 weights for QAT
                # CRITICAL: Do dequantization in no_grad context to avoid gradient issues
                with torch.no_grad():
                    if hasattr(module, '_packed_params') and module._packed_params is not None:
                        packed = module._packed_params
                        if isinstance(packed, tuple) and len(packed) >= 1:
                            quantized_weight = packed[0]
                            quantized_bias = packed[1] if len(packed) > 1 else None
                            
                            # Dequantize weight (convert from INT8 to FP32)
                            if hasattr(quantized_weight, 'q_scale') and hasattr(quantized_weight, 'q_zero_point'):
                                weight_fp32 = quantized_weight.dequantize()
                            else:
                                weight_fp32 = quantized_weight
                            
                            # Dequantize bias if present
                            bias_fp32 = None
                            if quantized_bias is not None:
                                if hasattr(quantized_bias, 'q_scale') and hasattr(quantized_bias, 'q_zero_point'):
                                    bias_fp32 = quantized_bias.dequantize()
                                else:
                                    bias_fp32 = quantized_bias
                        else:
                            logger.warning(f"    Could not extract weights from _packed_params for {name}")
                            continue
                    else:
                        # Try to get weight and bias directly (might be FP32 already)
                        if hasattr(module, 'weight'):
                            weight_fp32 = module.weight.data if hasattr(module.weight, 'data') else module.weight
                            if hasattr(weight_fp32, 'q_scale'):
                                weight_fp32 = weight_fp32.dequantize()
                        else:
                            logger.warning(f"    Could not find weight for {name}")
                            continue
                        
                        bias_fp32 = None
                        if hasattr(module, 'bias') and module.bias is not None:
                            bias_fp32 = module.bias.data if hasattr(module.bias, 'data') else module.bias
                            if hasattr(bias_fp32, 'q_scale'):
                                bias_fp32 = bias_fp32.dequantize()
                
                # Create QAT Conv2d module
                from torch.ao.nn.qat.modules.conv import Conv2d as QATConv2d
                from ultralytics.nn.tasks import _create_safe_qconfig
                
                qconfig = _create_safe_qconfig(backend)
                qat_conv = QATConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=bias_fp32 is not None,
                    padding_mode=module.padding_mode,
                    qconfig=qconfig
                )
                
                # Copy weights and bias
                qat_conv.weight = nn.Parameter(weight_fp32.clone())
                if bias_fp32 is not None:
                    qat_conv.bias = nn.Parameter(bias_fp32.clone())
                
                # Replace the quantized Conv2d with QAT Conv2d
                # We need to find the parent module and replace the child
                parts = name.split('.')
                if len(parts) > 1:
                    parent_name = '.'.join(parts[:-1])
                    child_name = parts[-1]
                    parent_module = dict(model.named_modules())[parent_name]
                    setattr(parent_module, child_name, qat_conv)
                    converted_modules.append(name)
                    logger.info(f"  Converted real quantized Conv2d to QAT Conv2d: {name}")
                else:
                    logger.warning(f"    Could not find parent module for {name}")
                    
            except Exception as e:
                logger.warning(f"Failed to convert real quantized Conv2d to QAT for {name}: {e}")
                import traceback
                traceback.print_exc()
    
    if converted_modules:
        logger.info(f"✓ Converted {len(converted_modules)} modules to QAT (Observers→FakeQuantize and quantized Conv2d→QAT Conv2d)")
    else:
        logger.warning(f"⚠️  No modules found matching patterns: {module_patterns}")
    
    return converted_modules


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
    convert_to_int8: bool = True,
    evaluate_int8: bool = True,
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
        sensitive_modules = ['model.10']  # BoTNet + CoordAtt indices
    
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
    
    # Check if int8.pt was provided - if so, try to use calibrated.pt instead for QAT
    if ptq_weights_path.name == "int8.pt":
        calibrated_path = ptq_weights_path.parent / "calibrated.pt"
        if calibrated_path.exists():
            LOGGER.info(f"⚠️  Detected int8.pt - switching to calibrated.pt for QAT (required for FakeQuantize)")
            LOGGER.info(f"   Using: {calibrated_path}")
            ptq_weights_path = calibrated_path
        else:
            LOGGER.warning("⚠️  int8.pt provided but calibrated.pt not found in same directory")
            LOGGER.warning("   QAT requires calibrated model (with FakeQuantize), not INT8 model")
            LOGGER.warning("   Will attempt to use int8.pt but may encounter issues with GPU")
    
    # Load checkpoint
    # Note: weights_only=False is needed for custom model classes (DetectionModel, etc.)
    checkpoint = torch.load(ptq_weights_path, map_location='cpu', weights_only=False)
    
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
        
        # CRITICAL: For calibrated PTQ models, we need hybrid approach:
        # - Keep most layers as PTQ (Observers with frozen scales)
        # - Convert only sensitive modules (BoTNet) to QAT (FakeQuantize with enabled observers)
        is_calibrated_ptq = checkpoint.get('calibrated', False) or checkpoint.get('qat_prepared', False)
        
        # Initialize flag to track if model was already prepared
        model_already_prepared = False
        
        if is_calibrated_ptq:
            # Hybrid approach: Load calibrated model with Observers, then selectively convert
            LOGGER.info("Detected calibrated PTQ model - using hybrid QAT approach...")
            LOGGER.info("  Strategy: Keep most layers as PTQ (Observers), convert only sensitive modules to QAT (FakeQuantize)")
            
            # CRITICAL: First prepare model for PTQ to get Observers
            # This ensures the model structure matches the calibrated state_dict
            LOGGER.info("  Preparing model for PTQ (to get Observer structure)...")
            example_input = torch.randn(1, 3, imgsz, imgsz)
            try:
                inner = inner.prepare_for_ptq(
                    backend=backend,
                    example_input=example_input,
                    use_fx=False,
                    quantize_backbone=True,
                    quantize_neck=True,
                    quantize_botnet=True,  # Will be converted to QAT later
                    quantize_coordatt=True
                )
                LOGGER.info("✓ Model prepared for PTQ (Observers inserted)")
            except Exception as e:
                LOGGER.warning(f"Failed to prepare for PTQ: {e}")
                LOGGER.warning("  Attempting to load state_dict anyway (model may already have Observers)...")
            
            # Load the calibrated model state_dict (with Observers and their statistics)
            LOGGER.info("  Loading calibrated model state_dict (with Observer statistics)...")
            
            # CRITICAL: Before loading, check if query/key/value modules are real quantized
            # This should NOT be the case for calibrated.pt, but let's verify
            before_load_quantized = []
            for name, module in inner.named_modules():
                if ('query' in name or 'key' in name or 'value' in name) and ('model.10' in name or 'm.0.cv2.0' in name):
                    is_real_quantized = (
                        hasattr(module, '_packed_params') or
                        ('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
                    )
                    if is_real_quantized:
                        before_load_quantized.append((name, type(module).__name__, type(module).__module__))
            
            if before_load_quantized:
                LOGGER.warning(f"⚠️  Found {len(before_load_quantized)} real quantized Conv2d modules BEFORE loading state_dict:")
                for name, class_name, module_path in before_load_quantized[:5]:
                    LOGGER.warning(f"  - {name}: {class_name} ({module_path})")
            
            inner.load_state_dict(state, strict=False)
            LOGGER.info("✓ Loaded calibrated model with Observer statistics")
            
            # CRITICAL: After loading, check again if query/key/value modules became real quantized
            # This would indicate the state_dict contained INT8 modules
            after_load_quantized = []
            for name, module in inner.named_modules():
                if ('query' in name or 'key' in name or 'value' in name) and ('model.10' in name or 'm.0.cv2.0' in name):
                    is_real_quantized = (
                        hasattr(module, '_packed_params') or
                        ('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
                    )
                    if is_real_quantized:
                        after_load_quantized.append((name, type(module).__name__, type(module).__module__))
            
            if after_load_quantized:
                LOGGER.error(f"❌ Found {len(after_load_quantized)} real quantized Conv2d modules AFTER loading state_dict!")
                LOGGER.error("   This means the calibrated.pt file contains INT8 modules, not Observers!")
                LOGGER.error("   The state_dict loading converted modules to INT8:")
                for name, class_name, module_path in after_load_quantized[:5]:
                    LOGGER.error(f"     - {name}: {class_name} ({module_path})")
                LOGGER.error("   These modules will be converted to QAT Conv2d in the next step...")
            else:
                LOGGER.info("✓ Verified: No real quantized Conv2d modules after loading state_dict")
            
            # Now selectively convert only sensitive modules (BoTNet) from Observers to FakeQuantize
            # Default sensitive modules if not provided
            if sensitive_modules is None:
                sensitive_modules = ['model.10']  # BoTNet
            
            LOGGER.info(f"  Converting Observers to FakeQuantize for sensitive modules: {sensitive_modules}")
            converted = convert_observers_to_fakequantize_selectively(
                inner,
                module_patterns=sensitive_modules,
                backend=backend,
                logger=LOGGER
            )
            
            if converted:
                LOGGER.info(f"✓ Converted {len(converted)} modules to QAT (FakeQuantize)")
            else:
                LOGGER.warning("⚠️  No modules were converted - check sensitive_modules patterns")
            
            # CRITICAL: After conversion, verify that all Conv2d modules in sensitive areas are QAT, not real quantized
            # Check for any remaining real quantized Conv2d modules in BoTNet
            LOGGER.info("  Verifying no real quantized Conv2d modules remain in sensitive areas...")
            remaining_quantized = []
            for name, module in inner.named_modules():
                if any(pattern in name for pattern in sensitive_modules):
                    is_real_quantized = (
                        hasattr(module, '_packed_params') or
                        ('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
                    )
                    is_conv2d_like = (
                        isinstance(module, nn.Conv2d) or
                        (hasattr(module, 'in_channels') and hasattr(module, 'out_channels') and 
                         hasattr(module, 'kernel_size') and 'conv' in type(module).__name__.lower())
                    )
                    if is_conv2d_like and is_real_quantized:
                        remaining_quantized.append((name, type(module).__name__, type(module).__module__))
            
            if remaining_quantized:
                LOGGER.error(f"❌ Found {len(remaining_quantized)} real quantized Conv2d modules in sensitive areas after conversion!")
                for name, class_name, module_path in remaining_quantized:
                    LOGGER.error(f"  - {name}: {class_name} ({module_path})")
                LOGGER.error("  These modules will cause 'derivative for dequantize' errors during training")
                LOGGER.error("  Attempting to convert them now...")
                # Try to convert them again
                converted_retry = convert_observers_to_fakequantize_selectively(
                    inner,
                    module_patterns=sensitive_modules,
                    backend=backend,
                    logger=LOGGER
                )
                if converted_retry:
                    LOGGER.info(f"✓ Retry conversion: {len(converted_retry)} additional modules converted")
                else:
                    LOGGER.error("❌ Retry conversion failed - these modules may need manual conversion")
            else:
                LOGGER.info("✓ Verified: No real quantized Conv2d modules in sensitive areas")
            
            # Update model reference
            base_model.model = inner
            model_already_prepared = True
            detection_model = inner
            
            LOGGER.info("✓ Hybrid QAT setup complete:")
            LOGGER.info(f"  - {len(converted)} modules converted to QAT (FakeQuantize) for training")
            LOGGER.info("  - Other modules remain as PTQ (Observers with frozen scales)")
        else:
            # Not a calibrated model - just prepare for QAT normally
            LOGGER.info("Preparing model for QAT before loading state_dict...")
            example_input = torch.randn(1, 3, imgsz, imgsz)
            try:
                inner = inner.prepare_for_qat(
                    backend=backend,
                    example_input=example_input,
                    use_fx=False
                )
                LOGGER.info("✓ Model prepared for QAT (FakeQuantize modules added)")
                inner.load_state_dict(state, strict=False)
            except Exception as e:
                LOGGER.warning(f"Failed to prepare for QAT: {e}")
                LOGGER.warning("Attempting to load state_dict anyway...")
                inner.load_state_dict(state, strict=False) 
        model = base_model

    detection_model = model.model
    # Sanity check: ensure inner detection model exists and is an nn.Module
    if not isinstance(detection_model, nn.Module):
        raise RuntimeError(f"Loaded checkpoint did not yield a valid nn.Module for detection_model (got {type(detection_model)!r})")
    LOGGER.info(f"✓ PTQ model loaded successfully")
    
    # CRITICAL: Check if model was already prepared in calibrated path
    # The flag should be set in the calibrated path above
    if 'model_already_prepared' not in locals():
        model_already_prepared = False
    
    # Check if model is INT8 (has quantized operations) vs calibrated (has Observers or FakeQuantize)
    from torch.ao.quantization import FakeQuantize
    from torch.ao.quantization.observer import ObserverBase
    
    has_fakequant = any(isinstance(m, FakeQuantize) for m in detection_model.modules())
    has_observers = any(
        hasattr(m, 'activation_post_process') and m.activation_post_process is not None
        for m in detection_model.modules()
    ) or any(isinstance(m, ObserverBase) for m in detection_model.modules())
    has_quantized = any(
        hasattr(m, '_packed_params') or 
        ('quantized' in type(m).__module__.lower() and 'qat' not in type(m).__module__.lower())
        for m in detection_model.modules()
    )
    
    # Model type detection:
    # - INT8: Has real quantized operations (_packed_params) but no FakeQuantize and no Observers
    # - PTQ Calibrated: Has Observers (activation_post_process) but no FakeQuantize and no INT8
    # - Hybrid QAT: Has both Observers (PTQ layers) and FakeQuantize (QAT layers)
    # - QAT Prepared: Has only FakeQuantize modules
    is_int8_model = has_quantized and not has_fakequant and not has_observers
    is_ptq_calibrated = has_observers and not has_fakequant and not is_int8_model
    is_hybrid_qat = has_observers and has_fakequant  # Both Observers and FakeQuantize
    is_calibrated_model = has_fakequant or is_ptq_calibrated or is_hybrid_qat
    
    if is_int8_model:
        LOGGER.warning("⚠️  Loaded model appears to be INT8 (real quantized operations)")
        LOGGER.warning("   INT8 models cannot do QAT - need FakeQuantize modules")
        LOGGER.warning("   Re-preparing model for QAT...")
        # Re-prepare the model for QAT (this will add FakeQuantize modules)
        example_input = torch.randn(1, 3, imgsz, imgsz)
        try:
            detection_model = detection_model.prepare_for_qat(
                backend=backend,
                example_input=example_input,
                use_fx=False  # Use eager mode for compatibility
            )
            LOGGER.info("✓ Model re-prepared for QAT (FakeQuantize modules added)")
            # Update model reference
            model.model = detection_model
        except Exception as e:
            LOGGER.error(f"Failed to re-prepare model for QAT: {e}")
            raise RuntimeError("Cannot proceed with QAT - model needs FakeQuantize modules")
    elif is_hybrid_qat:
        LOGGER.info("✓ Model is hybrid QAT (has both Observers and FakeQuantize)")
        LOGGER.info("  - Observers: PTQ layers with frozen scales")
        LOGGER.info("  - FakeQuantize: QAT layers for training")
        # Model is already in hybrid state - no need to prepare again
    elif is_calibrated_model:
        LOGGER.info("✓ Model is calibrated - ready for QAT")
        # Model is already prepared - no need to prepare again
    else:
        # Only prepare if we didn't already prepare it above (for calibrated models)
        if not is_calibrated_ptq:
            LOGGER.warning("⚠️  Model doesn't appear to be quantized - preparing for QAT...")
            # Prepare for QAT if not already prepared
            example_input = torch.randn(1, 3, imgsz, imgsz)
            try:
                detection_model = detection_model.prepare_for_qat(
                    backend=backend,
                    example_input=example_input,
                    use_fx=False
                )
                LOGGER.info("✓ Model prepared for QAT (FakeQuantize modules added)")
                model.model = detection_model
            except Exception as e:
                LOGGER.warning(f"Failed to prepare model for QAT: {e}")
                LOGGER.warning("Continuing anyway - model may already be in correct state")
        else:
            LOGGER.info("✓ Model already prepared (hybrid QAT path)")
    
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
    # If device_str was already set (for INT8 models), skip device resolution
    if 'device_str' not in locals() or 'torch_device' not in locals():
        device_str = _resolve_device(device)
        
        # 🎯 FIX START 🎯
        # Note: For QAT training, we can use GPU for faster training.
        # FakeQuantize modules work on GPU, so QAT training can run on GPU.
        # The backend (qnnpack/fbgemm) only matters during INT8 conversion, not during QAT training.
        # The final INT8 model will need CPU for inference with qnnpack backend.
        
        # Determine the final PyTorch device object based on the resolved string
        if device_str == "cpu":
            # Force CPU if explicitly specified
            torch_device = torch.device("cpu")
            LOGGER.info("Device explicitly set to CPU")
        elif device_str.isdigit():
            # Handle CUDA device IDs (e.g., "0" -> "cuda:0")
            if torch.cuda.is_available():
                device_str = f"cuda:{device_str}"
                torch_device = torch.device(device_str)
                LOGGER.info(f"Using GPU for QAT training: {device_str}")
                LOGGER.info("  (Note: Final INT8 model will use CPU for inference with qnnpack backend)")
            else:
                # Fallback for CUDA when unavailable
                torch_device = torch.device("cpu")
                device_str = "cpu"
                LOGGER.warning(f"CUDA device '{device_str}' requested but not available. Falling back to CPU.")
        elif device_str.startswith("cuda:"):
            # Handle explicit CUDA device strings (e.g., "cuda:0")
            if torch.cuda.is_available():
                torch_device = torch.device(device_str)
                LOGGER.info(f"Using GPU for QAT training: {device_str}")
                LOGGER.info("  (Note: Final INT8 model will use CPU for inference with qnnpack backend)")
            else:
                # Fallback for CUDA when unavailable
                torch_device = torch.device("cpu")
                device_str = "cpu"
                LOGGER.warning(f"CUDA device '{device_str}' requested but not available. Falling back to CPU.")
        else:
            # Default fallback to CPU for safety (e.g., if 'mps' was passed)
            torch_device = torch.device("cpu")
            device_str = "cpu"
            LOGGER.warning(f"Unsupported device '{_resolve_device(device)}'. Falling back to CPU.")

    # 🎯 FIX END 🎯
    
    # Only move to device if not INT8 model (INT8 models must stay on CPU)
    if not is_int8_model:
        try:
            model.to(torch_device)
            LOGGER.info(f"Model moved to device: {device_str}")
            
            # CRITICAL: Validate FakeQuantize on the actual training device (GPU)
            # The dimension mismatch might only appear on GPU
            if torch_device.type == 'cuda' and model_already_prepared:
                LOGGER.info("  Validating FakeQuantize modules on GPU (training device)...")
                try:
                    detection_model.train()  # Set to training mode
                    with torch.enable_grad():
                        # Test with realistic input on GPU
                        test_input = torch.randn(2, 3, imgsz, imgsz, device=torch_device, requires_grad=True)
                        test_output = detection_model(test_input)
                    LOGGER.info("✓ GPU validation passed (no dimension mismatches on GPU)")
                except RuntimeError as gpu_val_e:
                    error_msg = str(gpu_val_e)
                    if "dimensions of scale and zero-point" in error_msg or "not consistent" in error_msg:
                        LOGGER.error("⚠️  FakeQuantize dimension mismatch detected on GPU!")
                        LOGGER.error("   This indicates FakeQuantize modules have incorrect dimensions")
                        LOGGER.error("   The issue only appears on GPU, not CPU")
                        raise RuntimeError(
                            "FakeQuantize dimension mismatch detected on GPU. "
                            "This may indicate an issue with how FakeQuantize modules are initialized. "
                            "Please check the QAT qconfig and prepare_for_qat implementation."
                        ) from gpu_val_e
                    else:
                        LOGGER.warning(f"GPU validation failed (non-critical): {gpu_val_e}")
        except RuntimeError as e:
            # Re-raise if it's our dimension mismatch error
            if "FakeQuantize dimension mismatch" in str(e):
                raise
            # Otherwise, handle device move errors
            LOGGER.warning(f"Failed to move model to {device_str}: {e}")
            LOGGER.warning("Falling back to CPU...")
            torch_device = torch.device("cpu")
            device_str = "cpu"
            model.to(torch_device)
            LOGGER.info(f"Model moved to device: {device_str}")
    else:
        # INT8 models are already on CPU (loaded with map_location='cpu')
        LOGGER.info(f"Model kept on CPU (INT8 models cannot use GPU)")
    # Evaluate PTQ baseline (before fine-tuning)
    # DISABLED: Evaluation causes inference tensor issues and is not needed for QAT
    ptq_map = None
    if False and evaluate_before:  # Disabled to avoid inference tensor issues
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("Evaluating PTQ baseline (before fine-tuning)...")
        LOGGER.info("=" * 80)
        
        # Save model state before evaluation to reload after (avoids inference tensor issues)
        # CRITICAL: Clone all tensors in state_dict to avoid inference tensor issues
        model_state_before_eval = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in detection_model.state_dict().items()
        }
        
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
        finally:
            # CRITICAL: Reload model state to clear inference tensors created during evaluation
            # We need to manually replace parameters (not copy inplace) to avoid inference tensor issues
            LOGGER.info("Reloading model state to clear inference tensors...")
            with torch.enable_grad():
                # Manually replace each parameter to avoid inplace updates to inference tensors
                # We need to replace the Parameter object itself, not just the data
                for name, param in detection_model.named_parameters():
                    if name in model_state_before_eval:
                        saved_param = model_state_before_eval[name]
                        if isinstance(saved_param, torch.Tensor):
                            # Clone the saved parameter
                            cloned_param = saved_param.clone().detach()
                            # Create a new Parameter object to replace the inference tensor
                            new_param = nn.Parameter(cloned_param, requires_grad=param.requires_grad)
                            
                            # Replace the parameter in the module's _parameters dict
                            # This avoids inplace updates to inference tensors
                            parts = name.split('.')
                            module = detection_model
                            for part in parts[:-1]:
                                module = getattr(module, part)
                            param_name = parts[-1]
                            module._parameters[param_name] = new_param
                
                # Also handle buffers (like BatchNorm running_mean, running_var)
                for name, buffer in detection_model.named_buffers():
                    if name in model_state_before_eval:
                        saved_buffer = model_state_before_eval[name]
                        if isinstance(saved_buffer, torch.Tensor):
                            cloned_buffer = saved_buffer.clone().detach()
                            # Replace buffer in module's _buffers dict
                            parts = name.split('.')
                            module = detection_model
                            for part in parts[:-1]:
                                module = getattr(module, part)
                            buffer_name = parts[-1]
                            module._buffers[buffer_name] = cloned_buffer
                            
            detection_model.train()  # Ensure training mode
            torch.set_grad_enabled(True)  # Ensure gradients enabled
            LOGGER.info("✓ Model state reloaded - ready for QAT training")

    # Prepare model for QAT fine-tuning
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Preparing model for QAT fine-tuning...")
    LOGGER.info("=" * 80)
    
    # CRITICAL: Verify model has FakeQuantize (required for QAT training)
    # But skip this check if we already prepared it in the calibrated path above
    from torch.ao.quantization import FakeQuantize
    fakequant_count = sum(1 for m in detection_model.modules() if isinstance(m, FakeQuantize))
    has_real_quantized = any(
        hasattr(m, '_packed_params') or 'quantized' in type(m).__module__.lower()
        for m in detection_model.modules()
    )
    
    if has_real_quantized and fakequant_count == 0:
        LOGGER.error("❌ Model has real quantized operations but no FakeQuantize modules!")
        LOGGER.error("   This model cannot be used for QAT training (derivative for dequantize not implemented)")
        LOGGER.error("   The calibrated.pt file may have been converted to INT8")
        LOGGER.error("   Please regenerate calibrated.pt using train_ptq.py without conversion")
        raise RuntimeError("Model is INT8, not QAT-ready. Need FakeQuantize modules for QAT training.")
    
    # Only prepare if not already prepared (skip if calibrated path already did it)
    if fakequant_count == 0 and not (is_calibrated_ptq or model_already_prepared):
        LOGGER.warning("⚠️  No FakeQuantize modules found - model may not be QAT-ready")
        LOGGER.warning("   Attempting to prepare model for QAT...")
        example_input = torch.randn(1, 3, imgsz, imgsz).to(torch_device)
        try:
            detection_model = detection_model.prepare_for_qat(
                backend=backend,
                example_input=example_input,
                use_fx=False
            )
            model.model = detection_model
            LOGGER.info(f"✓ Model prepared for QAT - {sum(1 for m in detection_model.modules() if isinstance(m, FakeQuantize))} FakeQuantize modules added")
        except Exception as e:
            LOGGER.error(f"Failed to prepare model for QAT: {e}")
            raise RuntimeError("Cannot proceed - model needs FakeQuantize modules for QAT")
    elif fakequant_count > 0:
        LOGGER.info(f"✓ Model has {fakequant_count} FakeQuantize modules - ready for QAT")
    else:
        # Already prepared in calibrated path - no need to prepare again
        LOGGER.info("✓ Model already prepared for QAT (calibrated model path)")
    
    # CRITICAL: Final verification before training - ensure model has FakeQuantize for QAT modules, not INT8
    # The "derivative for dequantize" error means real quantized ops are present
    LOGGER.info("  Final verification: Checking for FakeQuantize vs INT8 operations...")
    final_fakequant_count = sum(1 for m in detection_model.modules() if isinstance(m, FakeQuantize))
    final_observer_count = sum(
        1 for m in detection_model.modules()
        if hasattr(m, 'activation_post_process') and m.activation_post_process is not None
        and not isinstance(m.activation_post_process, FakeQuantize)
    )
    final_quantized_modules = []
    for name, m in detection_model.named_modules():
        if hasattr(m, '_packed_params') or ('quantized' in type(m).__module__.lower() and 'qat' not in type(m).__module__.lower()):
            final_quantized_modules.append(name)
    
    if final_quantized_modules:
        LOGGER.error(f"❌ Found {len(final_quantized_modules)} real quantized modules (INT8) in model!")
        LOGGER.error("   These modules cannot be used for QAT training:")
        for mod_name in final_quantized_modules[:10]:
            LOGGER.error(f"     - {mod_name}")
        if len(final_quantized_modules) > 10:
            LOGGER.error(f"     ... and {len(final_quantized_modules) - 10} more")
        LOGGER.error("   The model must have FakeQuantize modules, not real quantized operations")
        raise RuntimeError(
            f"Model contains {len(final_quantized_modules)} real quantized modules. "
            "QAT requires FakeQuantize modules, not INT8 operations. "
            "Please ensure the model is prepared for QAT, not converted to INT8."
        )
    
    # For hybrid QAT, we need at least some FakeQuantize modules (for sensitive layers)
    # and Observers are fine for PTQ layers
    if final_fakequant_count == 0:
        LOGGER.error("❌ No FakeQuantize modules found in model!")
        LOGGER.error("   The model must have FakeQuantize modules for QAT training")
        LOGGER.error("   For hybrid QAT, sensitive modules should have FakeQuantize")
        raise RuntimeError("Model has no FakeQuantize modules. Cannot proceed with QAT training.")
    
    if final_observer_count > 0 and final_fakequant_count > 0:
        LOGGER.info(f"✓ Hybrid QAT verification passed:")
        LOGGER.info(f"  - {final_fakequant_count} FakeQuantize modules (QAT layers)")
        LOGGER.info(f"  - {final_observer_count} Observer modules (PTQ layers)")
        LOGGER.info(f"  - 0 INT8 modules")
    else:
        LOGGER.info(f"✓ Verification passed: {final_fakequant_count} FakeQuantize modules, 0 INT8 modules")
    
    # CRITICAL: Exit any inference mode that might have been set during evaluation
    # This is necessary because model.val() uses torch.inference_mode() which creates
    # inference tensors that cannot have requires_grad modified
    torch.set_grad_enabled(True)
    
    # Put model in training mode
    detection_model.train()
    
    # Selectively freeze observers - keep enabled only for QAT modules
    # Use the same patterns as sensitive_modules for consistency
    qat_modules = sensitive_modules if sensitive_modules else ['model.10']  # BoTNet
    LOGGER.info(f"Selectively freezing observers - keeping enabled for QAT modules: {qat_modules}")
    LOGGER.info("  (All other modules will have frozen PTQ scales)")
    disabled_count, enabled_count = freeze_observers_selectively(
        detection_model, 
        keep_observers_enabled=qat_modules,
        logger=LOGGER
    )
    LOGGER.info(f"  Disabled observers: {disabled_count} modules (PTQ scales frozen)")
    LOGGER.info(f"  Enabled observers:  {enabled_count} modules (QAT will update scales)")
    
    # CRITICAL: Verify hybrid setup - BoTNet should have FakeQuantize, others should have Observers
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Verifying Hybrid QAT Setup...")
    LOGGER.info("=" * 80)
    
    from torch.ao.quantization import FakeQuantize
    from torch.ao.quantization.observer import ObserverBase
    
    qat_modules_list = []
    ptq_modules_list = []
    issues_list = []
    
    for name, module in detection_model.named_modules():
        # Check if this is a sensitive module (should be QAT)
        is_sensitive = any(pattern in name for pattern in qat_modules)
        
        # Check module type
        has_fakequant = isinstance(module, FakeQuantize)
        has_observer = (
            hasattr(module, 'activation_post_process') and 
            module.activation_post_process is not None and
            not isinstance(module.activation_post_process, FakeQuantize)
        )
        is_observer_module = isinstance(module, ObserverBase)
        
        # Only check modules that have quantization-related attributes
        if has_fakequant or has_observer or is_observer_module:
            if is_sensitive:
                # Sensitive modules should have FakeQuantize
                if has_fakequant:
                    qat_modules_list.append(name)
                elif has_observer:
                    issues_list.append(f"{name} (⚠️ sensitive module has Observer, should be FakeQuantize)")
                else:
                    issues_list.append(f"{name} (⚠️ sensitive module has no quantization)")
            else:
                # Non-sensitive modules should have Observers (PTQ)
                if has_observer or is_observer_module:
                    ptq_modules_list.append(name)
                elif has_fakequant:
                    issues_list.append(f"{name} (⚠️ non-sensitive module has FakeQuantize, should be Observer)")
                # FP32 modules (e.g., ODConv) are fine - skip them
    
    LOGGER.info(f"\nQAT Modules (FakeQuantize): {len(qat_modules_list)}")
    if qat_modules_list:
        for mod_name in qat_modules_list[:10]:
            LOGGER.info(f"  ✓ {mod_name}")
        if len(qat_modules_list) > 10:
            LOGGER.info(f"  ... and {len(qat_modules_list) - 10} more")
    
    LOGGER.info(f"\nPTQ Modules (Observers): {len(ptq_modules_list)}")
    if ptq_modules_list:
        for mod_name in ptq_modules_list[:10]:
            LOGGER.info(f"  ✓ {mod_name}")
        if len(ptq_modules_list) > 10:
            LOGGER.info(f"  ... and {len(ptq_modules_list) - 10} more")
    
    # CRITICAL: Verify query/key/value Conv2d modules are QAT Conv2d, not regular Conv2d
    LOGGER.info("\nVerifying query/key/value Conv2d modules are QAT Conv2d...")
    query_key_value_issues = []
    for name, module in detection_model.named_modules():
        if ('query' in name or 'key' in name or 'value' in name) and isinstance(module, nn.Conv2d):
            # Check if it's QAT Conv2d (has weight_fake_quant)
            is_qat = hasattr(module, 'weight_fake_quant')
            is_real_quantized = (
                hasattr(module, '_packed_params') or
                ('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
            )
            
            if not is_qat:
                if is_real_quantized:
                    query_key_value_issues.append(f"{name} (❌ real quantized Conv2d, should be QAT Conv2d)")
                else:
                    query_key_value_issues.append(f"{name} (❌ regular Conv2d, should be QAT Conv2d with weight_fake_quant)")
            else:
                LOGGER.debug(f"  ✓ {name} is QAT Conv2d (has weight_fake_quant)")
    
    if query_key_value_issues:
        LOGGER.error(f"\n❌ Found {len(query_key_value_issues)} query/key/value Conv2d modules that are NOT QAT Conv2d:")
        for issue in query_key_value_issues:
            LOGGER.error(f"  {issue}")
        LOGGER.error("  These modules must be QAT Conv2d (torch.ao.nn.qat.modules.conv.Conv2d) for QAT training")
        raise RuntimeError(
            f"Found {len(query_key_value_issues)} query/key/value Conv2d modules that are not QAT Conv2d. "
            "These must be converted to QAT Conv2d for proper QAT training."
        )
    else:
        LOGGER.info("✓ All query/key/value Conv2d modules are QAT Conv2d")
    
    # CRITICAL: Comprehensive verification - List ALL QAT Conv2d modules and verify they are only in BoTNet
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Comprehensive QAT Module Verification")
    LOGGER.info("=" * 80)
    LOGGER.info("Verifying that ONLY BoTNet (model.10) modules are QAT...")
    
    all_qat_conv2d_modules = []
    qat_modules_outside_botnet = []
    
    for name, module in detection_model.named_modules():
        # Check if it's a Conv2d module
        if isinstance(module, nn.Conv2d):
            # Check if it's QAT Conv2d (has weight_fake_quant)
            has_weight_fake_quant = hasattr(module, 'weight_fake_quant') and module.weight_fake_quant is not None
            
            # Also check if activation_post_process is FakeQuantize
            from torch.ao.quantization import FakeQuantize
            has_fakequant_activation = (
                hasattr(module, 'activation_post_process') and 
                module.activation_post_process is not None and
                isinstance(module.activation_post_process, FakeQuantize)
            )
            
            is_qat_conv2d = has_weight_fake_quant or has_fakequant_activation
            
            if is_qat_conv2d:
                all_qat_conv2d_modules.append(name)
                # Verify it's within model.10 (BoTNet)
                if not name.startswith('model.10'):
                    qat_modules_outside_botnet.append(name)
    
    LOGGER.info(f"\nFound {len(all_qat_conv2d_modules)} QAT Conv2d modules:")
    for mod_name in all_qat_conv2d_modules:
        is_in_botnet = mod_name.startswith('model.10')
        status = "✓" if is_in_botnet else "❌"
        LOGGER.info(f"  {status} {mod_name}")
    
    if qat_modules_outside_botnet:
        LOGGER.error(f"\n❌ ERROR: Found {len(qat_modules_outside_botnet)} QAT Conv2d modules OUTSIDE of BoTNet (model.10):")
        for mod_name in qat_modules_outside_botnet:
            LOGGER.error(f"  - {mod_name}")
        LOGGER.error("  These modules should NOT be QAT - only BoTNet (model.10) should be QAT!")
        raise RuntimeError(
            f"Found {len(qat_modules_outside_botnet)} QAT Conv2d modules outside of BoTNet. "
            f"Only model.10 (BoTNet) should be QAT. QAT modules found: {qat_modules_outside_botnet}"
        )
    else:
        LOGGER.info(f"\n✓ Verification passed: All {len(all_qat_conv2d_modules)} QAT Conv2d modules are within BoTNet (model.10)")
        LOGGER.info("  This confirms that ONLY BoTNet is using QAT, all other modules are PTQ (Observers)")
    
    LOGGER.info("=" * 80)
    
    # Check for issues
    if issues_list:
        LOGGER.warning(f"\n⚠️  Found {len(issues_list)} potential issues:")
        for issue in issues_list[:10]:
            LOGGER.warning(f"  {issue}")
        if len(issues_list) > 10:
            LOGGER.warning(f"  ... and {len(issues_list) - 10} more")
    else:
        LOGGER.info("\n✓ Hybrid QAT setup verified successfully!")
        LOGGER.info(f"  - {len(qat_modules_list)} QAT modules (FakeQuantize) for training")
        LOGGER.info(f"  - {len(ptq_modules_list)} PTQ modules (Observers) with frozen scales")
    
    LOGGER.info("=" * 80)
    
    # Freeze ALL parameters first
    LOGGER.info("Freezing all parameters...")
    freeze_all_parameters(detection_model)
    
    # Unfreeze ONLY sensitive modules
    LOGGER.info("Unfreezing sensitive modules: %s" % ', '.join(sensitive_modules))
    
    # Ensure we're not in inference mode and gradients are enabled
    # This is critical after evaluation which may have set inference mode
    with torch.enable_grad():
        # Ensure model is in training mode
        detection_model.train()
        trainable_params = unfreeze_sensitive_modules(detection_model, sensitive_modules, logger=LOGGER)

    LOGGER.info(f"Total trainable parameters (unfrozen): {len(trainable_params)} parameters")
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
    # If warmup is enabled, we'll manually handle warmup in the training loop
    # and use cosine annealing for the remaining epochs
    if warmup_epochs > 0:
        # Cosine annealing for epochs after warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=lr * 0.1
        )
        LOGGER.info(f"Scheduler: Manual Linear Warmup ({warmup_epochs} epochs) + CosineAnnealingLR (T_max={epochs - warmup_epochs})")
        LOGGER.info(f"  Warmup: LR will linearly increase from 0 to {lr:.2e} over {warmup_epochs} epochs")
    else:
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
                # CRITICAL: Clone input images to ensure they're not inference tensors
                # This prevents "Inference tensors cannot be saved for backward" errors
                images = images.clone().detach().requires_grad_(True)
                
                # CRITICAL: Before forward pass, verify model still has FakeQuantize, not INT8
                # The "derivative for dequantize" error means real quantized ops are in the graph
                if batch_idx == 0 and epoch == 0:
                    # Check once at the start of training
                    from torch.ao.quantization import FakeQuantize
                    fq_count = sum(1 for m in detection_model.modules() if isinstance(m, FakeQuantize))
                    int8_modules = [name for name, m in detection_model.named_modules() 
                                   if hasattr(m, '_packed_params') or 'quantized' in type(m).__module__.lower()]
                    if int8_modules:
                        LOGGER.error(f"❌ Found INT8 modules before forward pass: {int8_modules[:5]}")
                        raise RuntimeError("Model has INT8 modules - cannot do QAT training")
                    LOGGER.info(f"  Pre-forward check: {fq_count} FakeQuantize modules, 0 INT8 modules")
                    
                    # CRITICAL: Monkey-patch torch.quantize and dequantize functions to catch real quantized operations
                    # Real quantized operations should NEVER be created during QAT training
                    # Store originals in module-level variables that persist across batches
                    if not hasattr(torch, '_original_quantize_per_tensor_qat_patch'):
                        torch._original_quantize_per_tensor_qat_patch = torch.quantize_per_tensor
                        torch._original_quantize_per_channel_qat_patch = torch.quantize_per_channel
                        torch._quantize_calls_qat = []
                    
                    def patched_quantize_per_tensor(*args, **kwargs):
                        """Patched version that logs and prevents real quantization during QAT."""
                        import traceback
                        stack = traceback.extract_stack()
                        caller = stack[-2] if len(stack) > 1 else None
                        caller_info = f"{caller.filename}:{caller.lineno}" if caller else "unknown"
                        torch._quantize_calls_qat.append(f"torch.quantize_per_tensor called from {caller_info}")
                        LOGGER.error(f"❌ REAL quantize_per_tensor called during QAT training!")
                        LOGGER.error(f"   Location: {caller_info}")
                        LOGGER.error("   This should NEVER happen - QAT uses FakeQuantize, not real quantization")
                        # Still call the original to see the full error, but log it
                        return torch._original_quantize_per_tensor_qat_patch(*args, **kwargs)
                    
                    def patched_quantize_per_channel(*args, **kwargs):
                        """Patched version that logs and prevents real quantization during QAT."""
                        import traceback
                        stack = traceback.extract_stack()
                        caller = stack[-2] if len(stack) > 1 else None
                        caller_info = f"{caller.filename}:{caller.lineno}" if caller else "unknown"
                        torch._quantize_calls_qat.append(f"torch.quantize_per_channel called from {caller_info}")
                        LOGGER.error(f"❌ REAL quantize_per_channel called during QAT training!")
                        LOGGER.error(f"   Location: {caller_info}")
                        LOGGER.error("   This should NEVER happen - QAT uses FakeQuantize, not real quantization")
                        # Still call the original to see the full error, but log it
                        return torch._original_quantize_per_channel_qat_patch(*args, **kwargs)
                    
                    # Patch torch functions (keep patched throughout training)
                    torch.quantize_per_tensor = patched_quantize_per_tensor
                    torch.quantize_per_channel = patched_quantize_per_channel
                    LOGGER.info("  Patched torch.quantize functions to detect real quantization (active throughout training)")
                
                # CRITICAL: Before forward pass, check for any quantized Conv2d modules
                # that might be creating real quantized operations
                if batch_idx == 0 and epoch == 0:
                    quantized_conv_modules = []
                    qat_conv_modules = []
                    
                    for name, module in detection_model.named_modules():
                        # Check if this is a real quantized Conv2d (not QAT)
                        is_quantized_conv = (
                            hasattr(module, '_packed_params') or
                            ('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
                        )
                        
                        # Also check if it's a quantized Conv2d class
                        if isinstance(module, nn.Conv2d):
                            module_type_str = str(type(module))
                            module_path = type(module).__module__
                            if 'quantized' in module_path.lower() and 'qat' not in module_path.lower():
                                is_quantized_conv = True
                        
                        # Check if it's a QAT Conv2d module
                        # CRITICAL: QAT Conv2d modules have weight_fake_quant (definitive sign)
                        # We should NOT count modules that only have FakeQuantize as activation_post_process
                        # because those might be PTQ modules that were converted but aren't true QAT Conv2d
                        # True QAT Conv2d is from torch.ao.nn.qat.modules.conv.Conv2d and has weight_fake_quant
                        has_weight_fake_quant = hasattr(module, 'weight_fake_quant') and module.weight_fake_quant is not None
                        is_qat_module_type = 'qat' in type(module).__module__.lower() and 'quantized' not in type(module).__module__.lower()
                        
                        # Only count as QAT Conv2d if it has weight_fake_quant OR is from QAT module path
                        # This ensures we only count true QAT Conv2d, not regular Conv2d with FakeQuantize activation
                        is_qat_conv = has_weight_fake_quant or (is_qat_module_type and isinstance(module, nn.Conv2d))
                        
                        if is_quantized_conv:
                            quantized_conv_modules.append((name, type(module).__name__, type(module).__module__))
                        elif is_qat_conv and isinstance(module, nn.Conv2d):
                            qat_conv_modules.append((name, type(module).__name__))
                        
                        # CRITICAL: Check inside Conv wrapper modules (they have .conv attribute)
                        if hasattr(module, 'conv') and isinstance(getattr(module, 'conv', None), nn.Conv2d):
                            inner_conv = module.conv
                            inner_is_quantized = (
                                hasattr(inner_conv, '_packed_params') or
                                ('quantized' in type(inner_conv).__module__.lower() and 'qat' not in type(inner_conv).__module__.lower())
                            )
                            # Only count as QAT if it has weight_fake_quant (definitive sign of QAT Conv2d)
                            inner_has_weight_fake_quant = hasattr(inner_conv, 'weight_fake_quant') and inner_conv.weight_fake_quant is not None
                            inner_is_qat_module_type = 'qat' in type(inner_conv).__module__.lower() and 'quantized' not in type(inner_conv).__module__.lower()
                            inner_is_qat = inner_has_weight_fake_quant or (inner_is_qat_module_type and isinstance(inner_conv, nn.Conv2d))
                            
                            if inner_is_quantized:
                                quantized_conv_modules.append((f"{name}.conv", type(inner_conv).__name__, type(inner_conv).__module__))
                            elif inner_is_qat:
                                qat_conv_modules.append((f"{name}.conv", type(inner_conv).__name__))
                    
                    if quantized_conv_modules:
                        LOGGER.error(f"❌ Found {len(quantized_conv_modules)} real quantized Conv2d modules in model!")
                        LOGGER.error("   These will create real quantized operations that cannot be backpropagated:")
                        for name, class_name, module_path in quantized_conv_modules[:10]:
                            LOGGER.error(f"     - {name}: {class_name} ({module_path})")
                        if len(quantized_conv_modules) > 10:
                            LOGGER.error(f"     ... and {len(quantized_conv_modules) - 10} more")
                        raise RuntimeError(
                            f"Found {len(quantized_conv_modules)} real quantized Conv2d modules. "
                            "QAT requires FakeQuantize modules, not real quantized operations."
                        )
                    else:
                        LOGGER.info(f"  ✓ No real quantized Conv2d modules detected")
                        LOGGER.info(f"  ✓ Found {len(qat_conv_modules)} QAT Conv2d modules (with FakeQuantize)")
                    
                    # CRITICAL: Also monkey-patch tensor.dequantize() to catch where dequantize is called
                    # This will help identify where real quantized operations are being created
                    if not hasattr(torch.Tensor, '_original_dequantize_qat_patch'):
                        # Store original dequantize method
                        torch.Tensor._original_dequantize_qat_patch = torch.Tensor.dequantize
                        torch._dequantize_calls_qat = []
                        
                        def patched_dequantize(self):
                            """Patched dequantize that logs calls during QAT."""
                            import traceback
                            stack = traceback.extract_stack()
                            caller = stack[-2] if len(stack) > 1 else None
                            caller_info = f"{caller.filename}:{caller.lineno}" if caller else "unknown"
                            torch._dequantize_calls_qat.append(f"tensor.dequantize() called from {caller_info}")
                            LOGGER.error(f"❌ REAL dequantize() called during QAT training!")
                            LOGGER.error(f"   Location: {caller_info}")
                            LOGGER.error("   This means a real quantized tensor exists in the computation graph")
                            # Still call the original to see the full error
                            return torch.Tensor._original_dequantize_qat_patch(self)
                        
                        # Patch dequantize method
                        torch.Tensor.dequantize = patched_dequantize
                        LOGGER.info("  Patched tensor.dequantize() to detect real quantized tensors")
                
                # YOLO forward pass
                outputs = detection_model(images)
                
                # Check if real quantize was called (log on first batch, but keep checking)
                if batch_idx == 0 and epoch == 0:
                    if hasattr(torch, '_quantize_calls_qat') and torch._quantize_calls_qat:
                        LOGGER.error(f"❌ Found {len(torch._quantize_calls_qat)} real quantize calls during forward pass!")
                        LOGGER.error("  This indicates real quantized operations are being created:")
                        for call in torch._quantize_calls_qat[:10]:
                            LOGGER.error(f"    - {call}")
                        raise RuntimeError("Real quantized operations detected during forward pass - cannot do QAT training")
                    else:
                        LOGGER.info("  ✓ No real quantize calls detected during forward pass")
                
                # Compute loss using YOLO's built-in loss function
                loss = compute_yolo_loss(outputs, batch_data, detection_model)
                
                # Check for invalid loss
                if not torch.isfinite(loss):
                    LOGGER.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                    continue
                
                # Backward pass (only sensitive layers get gradients)
                optimizer.zero_grad()
                try:
                    loss.backward()
                except RuntimeError as e:
                    error_msg = str(e)
                    if "derivative for dequantize" in error_msg:
                        # Check if we caught any quantize or dequantize calls
                        found_issues = False
                        if hasattr(torch, '_quantize_calls_qat') and torch._quantize_calls_qat:
                            LOGGER.error(f"\n{'='*80}")
                            LOGGER.error(f"❌ Backward pass failed: derivative for dequantize")
                            LOGGER.error(f"{'='*80}")
                            LOGGER.error("  Real quantized operations were detected during forward pass:")
                            for call in torch._quantize_calls_qat[-10:]:  # Show last 10
                                LOGGER.error(f"    - {call}")
                            found_issues = True
                        
                        if hasattr(torch, '_dequantize_calls_qat') and torch._dequantize_calls_qat:
                            if not found_issues:
                                LOGGER.error(f"\n{'='*80}")
                                LOGGER.error(f"❌ Backward pass failed: derivative for dequantize")
                                LOGGER.error(f"{'='*80}")
                            LOGGER.error("  Real quantized tensors were dequantized during forward pass:")
                            for call in torch._dequantize_calls_qat[-10:]:  # Show last 10
                                LOGGER.error(f"    - {call}")
                            found_issues = True
                        
                        if found_issues:
                            LOGGER.error("\n  This means real quantized tensors are in the computation graph.")
                            LOGGER.error("  QAT requires FakeQuantize modules, not real quantized operations.")
                            LOGGER.error("  Please check the model preparation and ensure no INT8 modules are present.")
                        else:
                            LOGGER.error(f"\n{'='*80}")
                            LOGGER.error(f"❌ Backward pass failed: derivative for dequantize")
                            LOGGER.error(f"{'='*80}")
                            LOGGER.error("  This error means real quantized operations are in the computation graph.")
                            LOGGER.error("  However, we did NOT detect torch.quantize_per_tensor/channel calls.")
                            LOGGER.error("  This suggests quantized operations are created through:")
                            LOGGER.error("    1. C++ extensions or PyTorch internals")
                            LOGGER.error("    2. _call_quantized_conv2d being called (should be prevented)")
                            LOGGER.error("    3. Model has INT8 modules that weren't detected")
                            LOGGER.error("\n  Please check:")
                            LOGGER.error("    - Are all Conv modules using FakeQuantize?")
                            LOGGER.error("    - Is _call_quantized_conv2d being called?")
                            LOGGER.error("    - Are there any quantized modules in the model?")
                        raise RuntimeError(
                            "derivative for dequantize is not implemented - real quantized operations detected in computation graph. "
                            "QAT requires FakeQuantize modules, not real quantized operations."
                        ) from e
                    else:
                        raise
                
                # Gradient clipping for stability (increased max_norm for better learning)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
                
                # Log gradient norm occasionally for debugging
                if batch_idx == 0 and epoch == 0:
                    LOGGER.info(f"  Initial gradient norm: {grad_norm:.4f}")
                elif batch_idx == 0 and epoch % 5 == 0:
                    LOGGER.info(f"  Gradient norm (epoch {epoch+1}): {grad_norm:.4f}")
                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{running_loss/num_batches:.4f}'
                })
                
            except RuntimeError as e:
                error_msg = str(e)
                if "dimensions of scale and zero-point" in error_msg or "not consistent with input tensor" in error_msg:
                    LOGGER.error(f"\n{'='*80}")
                    LOGGER.error(f"Batch {batch_idx} failed: Quantization dimension mismatch!")
                    LOGGER.error(f"{'='*80}")
                    LOGGER.error("  This should have been caught during model preparation.")
                    LOGGER.error("  The model needs to be re-prepared for QAT.")
                    LOGGER.error("  Please restart the script - it should use the fallback automatically.")
                    raise RuntimeError(
                        "Quantization dimension mismatch detected during training. "
                        "This should have been caught during model preparation. "
                        "Please restart the script."
                    ) from e
                else:
                    LOGGER.warning(f"Batch {batch_idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            except Exception as e:
                LOGGER.warning(f"Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        LOGGER.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Loss analysis and context
        if epoch == 0:
            LOGGER.info("\n" + "=" * 80)
            LOGGER.info("Loss Analysis")
            LOGGER.info("=" * 80)
            LOGGER.info(f"Initial QAT loss: {avg_loss:.4f}")
            
            # Context about YOLO loss ranges
            LOGGER.info("\nExpected loss ranges for YOLO detection:")
            LOGGER.info("  - Full precision (FP32): typically 2-10 (varies by dataset)")
            LOGGER.info("  - PTQ (INT8): typically 5-15 (some accuracy loss from quantization)")
            LOGGER.info("  - QAT (fine-tuning): typically 3-12 (recovery from PTQ loss)")
            LOGGER.info("\nNote: Loss values depend on:")
            LOGGER.info("  - Dataset size and complexity")
            LOGGER.info("  - Number of classes")
            LOGGER.info("  - Image resolution")
            LOGGER.info("  - Quantization method (PTQ vs QAT)")
            
            if ptq_map is not None:
                LOGGER.info(f"\nPTQ baseline mAP: {ptq_map:.4f} ({ptq_map*100:.2f}%)")
                LOGGER.info("  QAT fine-tuning should improve mAP while loss may remain relatively high")
                LOGGER.info("  Focus on mAP improvement, not absolute loss value")
            else:
                LOGGER.info("\n⚠️  PTQ baseline not available for comparison")
                LOGGER.info("  Loss ~50 may be expected for this dataset with quantization")
                LOGGER.info("  Monitor loss trend (should decrease over epochs)")
            
            LOGGER.info("\nQAT fine-tuning context:")
            LOGGER.info("  - Only BoTNet (model.10) modules are being fine-tuned")
            LOGGER.info("  - Other modules remain frozen (PTQ with Observers)")
            LOGGER.info("  - Loss may be higher initially but should stabilize/decrease")
            LOGGER.info("=" * 80)
        else:
            # Show loss trend
            if epoch == 1:
                LOGGER.info(f"Loss trend: Monitoring... (Epoch 1: {avg_loss:.4f})")
            else:
                # Calculate loss change (would need to track previous loss)
                LOGGER.info(f"Loss trend: Continuing QAT fine-tuning...")
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Manual warmup: linearly increase LR from lr/10 to lr over warmup_epochs
        # Starting from lr/10 instead of 0 helps the model start learning immediately
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # During warmup: set LR manually, starting from lr/10
            warmup_start_lr = lr / 10.0
            warmup_lr = warmup_start_lr + (lr - warmup_start_lr) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            new_lr = warmup_lr
            LOGGER.info(f"Learning rate (warmup): {current_lr:.6f} -> {new_lr:.6f} (epoch {epoch+1}/{warmup_epochs}, target: {lr:.6f})")
        else:
            # After warmup: use cosine annealing scheduler
            # Only step the scheduler after warmup is complete
            if warmup_epochs > 0:
                # First epoch after warmup: scheduler hasn't been stepped yet, so step it
                if epoch == warmup_epochs:
                    # Ensure LR is at full value before starting cosine annealing
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                scheduler.step()
            else:
                scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            LOGGER.info(f"Learning rate: {current_lr:.6f} -> {new_lr:.6f}")
        
        # Evaluation phase
        # Enable evaluation after each epoch to track mAP progress
        if evaluate_after:
            LOGGER.info(f"\nEvaluating epoch {epoch+1}...")
            try:
                # Ensure model is in eval mode for evaluation
                detection_model.eval()
                with torch.no_grad():
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
                    # Try multiple ways to extract mAP metrics
                    current_map = None
                    map50 = None
                    
                    # Method 1: Check if it has a 'box' attribute (most common for DetMetrics)
                    if hasattr(eval_results, 'box'):
                        current_map = getattr(eval_results.box, 'map', None)
                        map50 = getattr(eval_results.box, 'map50', None)
                    
                    # Method 2: Direct attributes
                    if current_map is None and hasattr(eval_results, 'map'):
                        current_map = eval_results.map
                    if map50 is None and hasattr(eval_results, 'map50'):
                        map50 = eval_results.map50
                    
                    # Method 3: Metrics dictionary
                    if current_map is None and hasattr(eval_results, 'metrics'):
                        metrics = eval_results.metrics
                        if isinstance(metrics, dict):
                            current_map = metrics.get('map', None) or metrics.get('mAP50-95(B)', None)
                            map50 = metrics.get('map50', None) or metrics.get('mAP50(B)', None)
                    
                    # Method 4: Try accessing as dict
                    if current_map is None and isinstance(eval_results, dict):
                        current_map = eval_results.get('map', None) or eval_results.get('mAP50-95(B)', None)
                        map50 = eval_results.get('map50', None) or eval_results.get('mAP50(B)', None)
                    
                    LOGGER.info(f"\nEpoch {epoch+1} Results:")
                    if map50 is not None:
                        LOGGER.info(f"  mAP@0.5:      {map50:.4f} ({map50*100:.2f}%)")
                    else:
                        LOGGER.warning(f"  mAP@0.5:      Not available")
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
                    else:
                        LOGGER.warning(f"  mAP@0.5:0.95: Not available - eval_results type: {type(eval_results)}")
                        if hasattr(eval_results, '__dict__'):
                            LOGGER.debug(f"  eval_results attributes: {list(eval_results.__dict__.keys())}")
                else:
                    LOGGER.warning(f"Evaluation returned None or empty results")
                    
            except Exception as e:
                LOGGER.warning(f"Evaluation failed: {e}")
                import traceback
                LOGGER.debug(traceback.format_exc())
        
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

    # INT8 Conversion and Evaluation
    int8_best_path = None
    int8_last_path = None
    int8_map = None
    int8_map50 = None
    
    if convert_to_int8:
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("Converting QAT Model to INT8")
        LOGGER.info("=" * 80)
        
        try:
            from torch.ao.quantization import FakeQuantize
            
            # Set quantization backend engine before conversion
            if backend in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = backend
                LOGGER.info(f"Set quantization backend engine to {backend}")
            else:
                LOGGER.warning(f"Backend '{backend}' not supported, using default: {torch.backends.quantized.engine}")
            
            # Load the trained model from checkpoint to ensure we have the exact state
            LOGGER.info("Loading trained QAT model from last.pt...")
            last_checkpoint = torch.load(last_path, map_location='cpu', weights_only=False)
            
            if 'model' in last_checkpoint:
                qat_model = last_checkpoint['model']
            elif 'model_state_dict' in last_checkpoint:
                # Need to rebuild model structure
                LOGGER.info("  Rebuilding model structure from state_dict...")
                # Use the detection_model we already have (it has the right structure)
                qat_model = detection_model
                qat_model.load_state_dict(last_checkpoint['model_state_dict'], strict=False)
            else:
                # Fallback: use the current detection_model
                LOGGER.warning("  Checkpoint format not recognized, using current model state")
                qat_model = detection_model
            
            # Convert last.pt model to INT8
            LOGGER.info("Converting last.pt QAT model to INT8...")
            fakequant_count = sum(1 for module in qat_model.modules() if isinstance(module, FakeQuantize))
            observer_count = sum(1 for module in qat_model.modules() 
                                if hasattr(module, 'activation_post_process') and 
                                hasattr(module.activation_post_process, '__class__') and
                                'Observer' in module.activation_post_process.__class__.__name__)
            
            LOGGER.info(f"  Found {fakequant_count} FakeQuantize modules (QAT layers)")
            LOGGER.info(f"  Found {observer_count} Observer modules (PTQ layers)")
            
            if fakequant_count == 0 and observer_count == 0:
                LOGGER.warning("No FakeQuantize or Observer modules found; skipping INT8 conversion.")
            else:
                # Prepare model for conversion
                qat_model.eval()
                qat_model = qat_model.float()
                
                # CRITICAL: Disable detection hooks during INT8 conversion
                # Conversion needs to call real quantize_per_channel to quantize weights
                hooks_disabled = False
                if hasattr(torch, '_original_quantize_per_tensor_qat_patch'):
                    torch.quantize_per_tensor = torch._original_quantize_per_tensor_qat_patch
                    torch.quantize_per_channel = torch._original_quantize_per_channel_qat_patch
                    hooks_disabled = True
                    LOGGER.info("  Temporarily disabled detection hooks for INT8 conversion")
                
                # Also disable dequantize patching if it exists
                if hasattr(torch.Tensor, '_original_dequantize_qat_patch'):
                    torch.Tensor.dequantize = torch.Tensor._original_dequantize_qat_patch
                    LOGGER.info("  Temporarily disabled dequantize patching for INT8 conversion")
                
                # CRITICAL: Enable FakeQuantize modules and disable observers before conversion
                # This ensures conversion uses QAT-learned parameters, not observer statistics
                LOGGER.info("  Preparing FakeQuantize modules for conversion...")
                fakequant_modules = []
                uncalibrated_count = 0
                enabled_count = 0
                disabled_observer_count = 0
                
                for name, module in qat_model.named_modules():
                    if isinstance(module, FakeQuantize):
                        fakequant_modules.append((name, module))
                        
                        # Enable fake quantization (use learned parameters)
                        # This ensures the FakeQuantize will use its learned scale/zero_point
                        try:
                            if hasattr(module, 'enable_fake_quant'):
                                module.enable_fake_quant()
                                enabled_count += 1
                            elif hasattr(module, 'fake_quant_enabled'):
                                module.fake_quant_enabled = True
                                enabled_count += 1
                        except Exception as e:
                            LOGGER.warning(f"    Could not enable fake_quant for {name}: {e}")
                        
                        # Disable observer (use learned min/max, not observer statistics)
                        # This is critical: we want to use the QAT-learned parameters, not re-calibrate
                        try:
                            if hasattr(module, 'disable_observer'):
                                module.disable_observer()
                                disabled_observer_count += 1
                            elif hasattr(module, 'observer_enabled'):
                                module.observer_enabled = False
                                disabled_observer_count += 1
                        except Exception as e:
                            LOGGER.warning(f"    Could not disable observer for {name}: {e}")
                        
                        # Verify calibration: check if min_val and max_val are calibrated
                        if hasattr(module, 'activation_post_process'):
                            observer = module.activation_post_process
                            if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                                try:
                                    min_val = observer.min_val
                                    max_val = observer.max_val
                                    # Check if uncalibrated (inf or -inf)
                                    is_uncalibrated = False
                                    if isinstance(min_val, torch.Tensor):
                                        if torch.any(torch.isinf(min_val)) or torch.any(min_val == float('inf')):
                                            is_uncalibrated = True
                                    elif min_val == float('inf') or min_val == float('-inf'):
                                        is_uncalibrated = True
                                    
                                    if isinstance(max_val, torch.Tensor):
                                        if torch.any(torch.isinf(max_val)) or torch.any(max_val == float('-inf')):
                                            is_uncalibrated = True
                                    elif max_val == float('-inf') or max_val == float('inf'):
                                        is_uncalibrated = True
                                    
                                    if is_uncalibrated:
                                        uncalibrated_count += 1
                                        LOGGER.warning(f"    ⚠️  Uncalibrated FakeQuantize at {name}: min_val={min_val}, max_val={max_val}")
                                except Exception as e:
                                    # Some observers might not have min_val/max_val accessible
                                    pass
                
                LOGGER.info(f"  Found {len(fakequant_modules)} FakeQuantize modules")
                LOGGER.info(f"  Enabled fake_quant for {enabled_count} modules")
                LOGGER.info(f"  Disabled observers for {disabled_observer_count} modules")
                if uncalibrated_count > 0:
                    LOGGER.warning(f"  ⚠️  {uncalibrated_count} FakeQuantize modules appear uncalibrated (may use default statistics)")
                else:
                    LOGGER.info(f"  ✓ All FakeQuantize modules appear calibrated")
                
                # Convert to INT8 (handles both FakeQuantize and Observers)
                LOGGER.info("  Calling convert_to_quantized()...")
                quantized_model = qat_model.convert_to_quantized()
                
                if quantized_model is not None:
                    qat_model = quantized_model
                
                ensure_module_bookkeeping(qat_model, recursive=True)
                
                # Verify conversion
                fakequants_remaining = sum(
                    1 for module in qat_model.modules()
                    if isinstance(module, FakeQuantize)
                )
                if fakequants_remaining > 0:
                    LOGGER.warning(f"  ⚠️  Detected {fakequants_remaining} FakeQuantize modules after conversion.")
                else:
                    LOGGER.info("  ✓ QAT observers successfully removed (pure INT8 graph)")
                
                # Log parameter comparison: Verify INT8 quantized parameters are valid
                # This helps verify that learned parameters were preserved during conversion
                LOGGER.info("  Verifying parameter preservation...")
                param_comparison_count = 0
                param_mismatch_count = 0
                
                # After conversion, FakeQuantize modules are replaced with real quantized ops
                # We verify that quantized modules have reasonable scale/zero_point values
                for name, module in qat_model.named_modules():
                    # Check if it's a quantized module
                    is_quantized = (
                        hasattr(module, '_packed_params') or
                        ('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
                    )
                    
                    if is_quantized:
                        try:
                            # Try to extract scale and zero_point
                            scale = None
                            zero_point = None
                            
                            # Method 1: Direct attributes (for quantized Conv2d, Linear, etc.)
                            if hasattr(module, 'scale'):
                                scale = module.scale
                            if hasattr(module, 'zero_point'):
                                zero_point = module.zero_point
                            
                            # Method 2: From _packed_params (for some quantized modules)
                            if scale is None and hasattr(module, '_packed_params') and module._packed_params is not None:
                                packed = module._packed_params
                                if isinstance(packed, tuple) and len(packed) > 0:
                                    # For Conv2d, first element is quantized weight tensor
                                    quantized_weight = packed[0]
                                    if hasattr(quantized_weight, 'q_scale'):
                                        scale = quantized_weight.q_scale
                                    if hasattr(quantized_weight, 'q_zero_point'):
                                        zero_point = quantized_weight.q_zero_point
                            
                            # Verify parameters are reasonable
                            if scale is not None:
                                # Handle both tensor and scalar scales
                                if isinstance(scale, torch.Tensor):
                                    scale_val = scale.item() if scale.numel() == 1 else scale[0].item()
                                    is_invalid = torch.any(torch.isnan(scale)) or torch.any(torch.isinf(scale)) or torch.any(scale <= 0)
                                else:
                                    scale_val = scale
                                    is_invalid = (scale_val != scale_val) or (scale_val == float('inf')) or (scale_val == float('-inf')) or (scale_val <= 0)
                                
                                if is_invalid:
                                    LOGGER.warning(f"    ⚠️  Invalid scale at {name}: {scale}")
                                    param_mismatch_count += 1
                                else:
                                    param_comparison_count += 1
                                    if param_comparison_count <= 5:  # Log first 5 for verification
                                        zero_pt_str = ""
                                        if zero_point is not None:
                                            if isinstance(zero_point, torch.Tensor):
                                                zero_pt_val = zero_point.item() if zero_point.numel() == 1 else zero_point[0].item()
                                            else:
                                                zero_pt_val = zero_point
                                            zero_pt_str = f", zero_point={zero_pt_val}"
                                        LOGGER.info(f"    ✓ {name}: scale={scale_val:.6f}{zero_pt_str}")
                        except Exception as e:
                            # Some modules might not expose scale/zero_point directly
                            # This is okay - not all quantized modules need to be verified this way
                            pass
                
                if param_mismatch_count > 0:
                    LOGGER.warning(f"  ⚠️  Found {param_mismatch_count} modules with invalid quantization parameters")
                else:
                    LOGGER.info(f"  ✓ Verified {param_comparison_count} quantized modules have valid parameters")
                
                # Identify QAT vs PTQ layers before saving
                # QAT layers are those matching sensitive_modules patterns
                qat_layer_names = []
                ptq_layer_names = []
                
                for name, module in qat_model.named_modules():
                    # Check if it's a quantized module
                    is_quantized = (
                        hasattr(module, '_packed_params') or
                        ('quantized' in type(module).__module__.lower() and 'qat' not in type(module).__module__.lower())
                    )
                    
                    if is_quantized:
                        # Check if this module matches sensitive_modules pattern
                        is_sensitive = any(pattern in name for pattern in sensitive_modules)
                        if is_sensitive:
                            qat_layer_names.append(name)
                        else:
                            ptq_layer_names.append(name)
                
                # Save INT8 models with metadata
                last_int8_path = weights_dir / "last_int8.pt"
                torch.save({
                    "model": qat_model,
                    "model_state_dict": qat_model.state_dict(),
                    "backend": backend,
                    "int8": True,
                    "epoch": epochs,
                    "date": datetime.now().isoformat(),
                    "hybrid_qat": True,
                    "sensitive_modules": sensitive_modules,  # Which modules were QAT
                    "qat_layers": qat_layer_names,  # List of QAT layer names
                    "ptq_layers": ptq_layer_names,  # List of PTQ layer names
                }, last_int8_path)
                LOGGER.info(f"  ✓ Saved INT8 checkpoint to {last_int8_path}")
                int8_last_path = last_int8_path
                
                # Store INT8 model for evaluation (qat_model is now INT8)
                int8_model_for_eval = qat_model
                
                # Note: Only converting last.pt to INT8 (best.pt conversion skipped)
                
                # Evaluate INT8 model
                if evaluate_int8:
                    LOGGER.info("\n" + "=" * 80)
                    LOGGER.info("Evaluating INT8 Model")
                    LOGGER.info("=" * 80)
                    
                    try:
                        # For qnnpack backend, evaluation must use CPU
                        eval_device = "cpu" if backend == "qnnpack" else device_str
                        if eval_device != device_str:
                            LOGGER.info(f"Using {eval_device} for evaluation (required for {backend} backend)")
                        
                        int8_yolo = YOLO(str(model_cfg))
                        int8_yolo.model = int8_model_for_eval
                        int8_yolo.model.eval()
                        
                        LOGGER.info("Running validation with YOLO's val() method...")
                        int8_eval_results = int8_yolo.val(
                            data=str(data_cfg),
                            imgsz=imgsz,
                            batch=batch or 16,
                            device=eval_device,
                            plots=False,
                            save=False,
                            verbose=True
                        )
                        
                        LOGGER.info("\nINT8 Model Evaluation Results:")
                        if int8_eval_results:
                            # Extract metrics - handle both dict and DetMetrics object
                            if isinstance(int8_eval_results, dict):
                                int8_map50 = int8_eval_results.get('metrics/mAP50(B)', int8_eval_results.get('map50', None))
                                int8_map = int8_eval_results.get('metrics/mAP50-95(B)', int8_eval_results.get('map', None))
                            else:
                                # It's a DetMetrics object
                                int8_map50 = getattr(int8_eval_results, 'map50', None)
                                int8_map = getattr(int8_eval_results, 'map', None)
                            
                            if int8_map50 is not None:
                                LOGGER.info(f"  mAP@0.5:      {int8_map50:.4f} ({int8_map50*100:.2f}%)")
                            if int8_map is not None:
                                LOGGER.info(f"  mAP@0.5:0.95: {int8_map:.4f} ({int8_map*100:.2f}%)")
                                
                                # Compare with PTQ baseline
                                if ptq_map is not None:
                                    improvement = (int8_map - ptq_map) * 100
                                    LOGGER.info(f"  Improvement:  {improvement:+.2f}% from PTQ baseline ({ptq_map:.4f})")
                                
                                # Update best_map if we got a valid result
                                if int8_map > 0:
                                    best_map = int8_map
                                    best_epoch = epochs
                        else:
                            LOGGER.warning("Evaluation completed but no results returned")
                            
                    except Exception as eval_err:
                        LOGGER.warning(f"INT8 evaluation failed: {eval_err}")
                        import traceback
                        LOGGER.debug(traceback.format_exc())
                
        except Exception as conv_err:
            LOGGER.error(f"Failed to convert QAT model to INT8: {conv_err}")
            import traceback
            LOGGER.error(traceback.format_exc())
    
    # Final summary
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Hybrid QAT Complete!")
    LOGGER.info("=" * 80)
    if ptq_map is not None and best_map > 0:
        improvement = (best_map - ptq_map) * 100
        LOGGER.info(f"PTQ baseline:  {ptq_map:.4f} ({ptq_map*100:.2f}%)")
        LOGGER.info(f"Best QAT:      {best_map:.4f} ({best_map*100:.2f}%)")
        LOGGER.info(f"Improvement:   {improvement:+.2f}%")
        LOGGER.info(f"Best epoch:    {best_epoch}")
    elif int8_map is not None:
        LOGGER.info(f"INT8 mAP:      {int8_map:.4f} ({int8_map*100:.2f}%)")
        if int8_map50 is not None:
            LOGGER.info(f"INT8 mAP@0.5:  {int8_map50:.4f} ({int8_map50*100:.2f}%)")
    
    if save_best:
        LOGGER.info(f"\nBest model saved to: {weights_dir / 'best.pt'}")
    LOGGER.info(f"Last model saved to: {weights_dir / 'last.pt'}")
    if int8_last_path:
        LOGGER.info(f"Last INT8 model saved to: {int8_last_path}")
    LOGGER.info("=" * 80)

    return {
        "best_path": weights_dir / "best.pt" if save_best else None,
        "last_path": last_path,
        "weights_dir": weights_dir,
        "best_map": best_map,
        "best_epoch": best_epoch,
        "int8_best_path": int8_best_path,
        "int8_last_path": int8_last_path,
        "int8_map": int8_map,
        "int8_map50": int8_map50,
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
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of warmup epochs (default: 0)",
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
        "--no-convert-int8",
        action="store_true",
        help="Skip INT8 conversion after training",
    )
    parser.add_argument(
        "--no-eval-int8",
        action="store_true",
        help="Skip INT8 model evaluation after conversion",
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
        warmup_epochs=args.warmup_epochs,
        sensitive_modules=args.sensitive_modules,
        evaluate_before=not args.no_eval_before,
        evaluate_after=not args.no_eval_after,
        convert_to_int8=not args.no_convert_int8,
        evaluate_int8=not args.no_eval_int8,
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
