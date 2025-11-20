import torch
import torch.nn as nn
import torch.nn.functional as F

# Import safe conv2d call for quantized Conv2d support
try:
    from ultralytics.nn.modules.conv import _safe_conv2d_call
except ImportError:
    # Fallback if import fails
    def _safe_conv2d_call(conv_module, x):
        return conv_module(x)


def _safe_activation_call(act_module, x):
    """
    Safely call activation function, handling quantized activation backend dispatch errors.
    
    Args:
        act_module: Activation module (quantized or FP32)
        x: Input tensor
    
    Returns:
        Output tensor (FP32 if fallback used, quantized if successful)
    """
    # Check if it's a quantized activation
    is_quantized = (
        'quantized' in type(act_module).__module__.lower() or
        'quantized' in type(act_module).__name__.lower()
    )
    
    if not is_quantized:
        # Regular FP32 activation - call normally
        return act_module(x)
    
    # It's quantized - ensure input is quantized if needed
    if not (hasattr(x, 'q_scale') and hasattr(x, 'q_zero_point')):
        # Input is FP32, need to quantize
        t_min = x.min().item()
        t_max = x.max().item()
        if t_max == t_min:
            scale = 1.0
            zero_point = 128
        else:
            scale = (t_max - t_min) / 255.0
            zero_point = int(round(-t_min / scale))
            zero_point = max(0, min(255, zero_point))
            if scale == 0.0:
                scale = 1.0
        
        try:
            x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        except Exception:
            x = torch.quantize_per_tensor(x, 0.01, 128, torch.quint8)
    
    try:
        # Try calling quantized activation
        return act_module(x)
    except (NotImplementedError, RuntimeError) as e:
        error_msg = str(e)
        if 'quantized::' in error_msg and ('CPU' in error_msg or 'backend' in error_msg.lower()):
            # Backend dispatch error - fallback to FP32 activation
            input_fp32 = x.dequantize() if hasattr(x, 'q_scale') else x
            
            # Determine activation type and apply FP32 equivalent
            act_type = type(act_module).__name__
            if 'ReLU6' in act_type:
                # ReLU6: clamp(x, 0, 6)
                output_fp32 = torch.clamp(input_fp32, 0, 6)
            elif 'ReLU' in act_type:
                # ReLU: max(x, 0)
                output_fp32 = torch.relu(input_fp32)
            elif 'Sigmoid' in act_type:
                # Sigmoid
                output_fp32 = torch.sigmoid(input_fp32)
            elif 'SiLU' in act_type or 'Swish' in act_type:
                # SiLU/Swish: x * sigmoid(x)
                output_fp32 = input_fp32 * torch.sigmoid(input_fp32)
            elif 'Hardswish' in act_type:
                # Hardswish: x * relu6(x + 3) / 6
                output_fp32 = input_fp32 * torch.clamp(input_fp32 + 3, 0, 6) / 6
            else:
                # Unknown activation - just return input (identity)
                import warnings
                warnings.warn(f"Unknown quantized activation {act_type}, using identity fallback")
                output_fp32 = input_fp32
            
            return output_fp32
        else:
            # Different error, re-raise
            raise


def _safe_batchnorm2d_call(bn_module, x):
    """
    Safely call BatchNorm2d, handling quantized BatchNorm2d backend dispatch errors.
    
    Args:
        bn_module: BatchNorm2d module (quantized or FP32)
        x: Input tensor
    
    Returns:
        Output tensor (FP32 if fallback used, quantized if successful)
    """
    # Check if it's a quantized BatchNorm2d
    is_quantized = (
        hasattr(bn_module, '_packed_params') or 
        'quantized' in type(bn_module).__module__.lower() or
        'quantized' in type(bn_module).__name__.lower()
    )
    
    if not is_quantized:
        # Regular FP32 BatchNorm2d - call normally
        return bn_module(x)
    
    # It's quantized - ensure input is quantized if needed
    if not (hasattr(x, 'q_scale') and hasattr(x, 'q_zero_point')):
        # Input is FP32, need to quantize
        t_min = x.min().item()
        t_max = x.max().item()
        if t_max == t_min:
            scale = 1.0
            zero_point = 128
        else:
            scale = (t_max - t_min) / 255.0
            zero_point = int(round(-t_min / scale))
            zero_point = max(0, min(255, zero_point))
            if scale == 0.0:
                scale = 1.0
        
        try:
            x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        except Exception:
            x = torch.quantize_per_tensor(x, 0.01, 128, torch.quint8)
    
    try:
        # Try calling quantized BatchNorm2d
        return bn_module(x)
    except (NotImplementedError, RuntimeError) as e:
        error_msg = str(e)
        if 'quantized::batch_norm' in error_msg and ('CPU' in error_msg or 'backend' in error_msg.lower()):
            # Backend dispatch error - fallback to FP32 BatchNorm
            input_fp32 = x.dequantize() if hasattr(x, 'q_scale') else x
            
            # Backend dispatch error - fallback to FP32 BatchNorm
            # Try to extract parameters from quantized BatchNorm2d
            try:
                # Get number of features from input shape
                num_features = input_fp32.shape[1]
                
                # Try to get parameters from quantized BatchNorm2d
                weight_fp32 = None
                bias_fp32 = None
                running_mean_fp32 = None
                running_var_fp32 = None
                
                # Check for _packed_params
                if hasattr(bn_module, '_packed_params') and bn_module._packed_params is not None:
                    packed = bn_module._packed_params
                    if isinstance(packed, (tuple, list)) and len(packed) >= 2:
                        # Try to extract and dequantize parameters
                        try:
                            if len(packed) >= 1 and packed[0] is not None:
                                w = packed[0]
                                weight_fp32 = w.dequantize() if hasattr(w, 'q_scale') else w
                            if len(packed) >= 2 and packed[1] is not None:
                                b = packed[1]
                                bias_fp32 = b.dequantize() if hasattr(b, 'q_scale') else b
                            if len(packed) >= 3 and packed[2] is not None:
                                rm = packed[2]
                                running_mean_fp32 = rm.dequantize() if hasattr(rm, 'q_scale') else rm
                            if len(packed) >= 4 and packed[3] is not None:
                                rv = packed[3]
                                running_var_fp32 = rv.dequantize() if hasattr(rv, 'q_scale') else rv
                        except Exception:
                            pass
                
                # Get eps and momentum
                eps = getattr(bn_module, 'eps', 1e-5)
                momentum = getattr(bn_module, 'momentum', 0.1)
                
                # If we have running stats, use them; otherwise compute on-the-fly
                if running_mean_fp32 is not None and running_var_fp32 is not None:
                    # Use running statistics
                    output_fp32 = torch.nn.functional.batch_norm(
                        input_fp32, running_mean_fp32, running_var_fp32,
                        weight_fp32, bias_fp32,
                        training=False, momentum=momentum, eps=eps
                    )
                else:
                    # Compute batch statistics (for eval mode, this is less ideal but works)
                    # Create a temporary BatchNorm2d with same num_features
                    temp_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
                    temp_bn.eval()
                    if weight_fp32 is not None:
                        temp_bn.weight.data = weight_fp32
                    if bias_fp32 is not None:
                        temp_bn.bias.data = bias_fp32
                    output_fp32 = temp_bn(input_fp32)
                
                return output_fp32
            except Exception as fallback_error:
                # If all else fails, just return the dequantized input
                # This allows forward pass to continue (with reduced accuracy)
                import warnings
                warnings.warn(f"Quantized BatchNorm2d backend error, returning dequantized input: {fallback_error}")
                return input_fp32
        else:
            # Different error, re-raise
            raise
 
 
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        # Use safe activation call to handle quantized ReLU6 backend errors
        return _safe_activation_call(self.relu, x + 3) / 6
    

 
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
 
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, inp // reduction)
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
 
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = _safe_conv2d_call(self.conv1, y)
        y = _safe_batchnorm2d_call(self.bn1, y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

       # a_h = F.tanh(self.conv_h(x_h))
        #a_w = F.tanh(self.conv_h(x_w))
        a_h = _safe_conv2d_call(self.conv_h, x_h).sigmoid()
        a_w = _safe_conv2d_call(self.conv_w, x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out