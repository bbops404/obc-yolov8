#!/usr/bin/env python3
"""Generate a simple list of all quantized layers."""

import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics.nn.tasks import ensure_module_bookkeeping

checkpoint = torch.load('/home/ubuntu/obc-yolov8/runs/detect/train_qat25/weights/last_int8.pt', map_location='cpu', weights_only=False)
model = checkpoint.get('model')
ensure_module_bookkeeping(model, recursive=True)

# Collect all quantized layers with their values
quantized_layers = []
for name, module in model.named_modules():
    module_type = type(module).__name__
    module_path = type(module).__module__
    
    # Check if it's a quantized Conv2d directly (not wrapped)
    # Quantized Conv2d modules are from torch.ao.nn.quantized, not torch.nn
    if module_type == 'Conv2d' and 'quantized' in module_path.lower():
        if not name.endswith('.conv'):  # Skip .conv children (handled separately)
            scale = module.scale.item() if hasattr(module, 'scale') and isinstance(module.scale, torch.Tensor) else (module.scale if hasattr(module, 'scale') else None)
            zero_point = module.zero_point.item() if hasattr(module, 'zero_point') and isinstance(module.zero_point, torch.Tensor) else (module.zero_point if hasattr(module, 'zero_point') else None)
            quantized_layers.append((name, scale, zero_point))
    
    # Check inside Conv wrappers
    if hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Module):
        conv_module = module.conv
        conv_path = type(conv_module).__module__
        conv_type = type(conv_module).__name__
        is_quantized = (
            hasattr(conv_module, '_packed_params') or 
            'quantized' in conv_path.lower() or
            'Quantized' in conv_type
        )
        if is_quantized:
            scale = conv_module.scale.item() if hasattr(conv_module, 'scale') and isinstance(conv_module.scale, torch.Tensor) else (conv_module.scale if hasattr(conv_module, 'scale') else None)
            zero_point = conv_module.zero_point.item() if hasattr(conv_module, 'zero_point') and isinstance(conv_module.zero_point, torch.Tensor) else (conv_module.zero_point if hasattr(conv_module, 'zero_point') else None)
            quantized_layers.append((f'{name}.conv', scale, zero_point))

# Remove duplicates and sort by layer name
seen = set()
unique_layers = []
for layer_name, scale, zero_point in quantized_layers:
    if layer_name not in seen:
        seen.add(layer_name)
        unique_layers.append((layer_name, scale, zero_point))
quantized_layers = sorted(unique_layers, key=lambda x: x[0])

# Group by type
direct_layers = [(name, scale, zp) for name, scale, zp in quantized_layers if not name.endswith('.conv')]
wrapped_layers = [(name, scale, zp) for name, scale, zp in quantized_layers if name.endswith('.conv')]

with open('quantized_layers_simple_list.txt', 'w') as f:
    f.write('='*80 + '\n')
    f.write('COMPLETE LIST OF QUANTIZED LAYERS WITH VALUES\n')
    f.write('='*80 + '\n\n')
    f.write(f'Total: {len(quantized_layers)} quantized Conv2d layers\n\n')
    
    f.write(f'DIRECT QUANTIZED CONV2D LAYERS (CoordAtt - {len(direct_layers)}):\n')
    f.write('-'*80 + '\n')
    if direct_layers:
        for i, (layer, scale, zp) in enumerate(direct_layers, 1):
            scale_str = f'{scale:.8f}' if scale is not None else 'N/A'
            zp_str = f'{zp}' if zp is not None else 'N/A'
            f.write(f'{i:3d}. {layer:40s}  scale={scale_str:12s}  zero_point={zp_str:4s}\n')
    else:
        f.write('  (none)\n')
    
    f.write(f'\n\nQUANTIZED CONV2D LAYERS INSIDE CONV WRAPPERS ({len(wrapped_layers)}):\n')
    f.write('-'*80 + '\n')
    for i, (layer, scale, zp) in enumerate(wrapped_layers, 1):
        scale_str = f'{scale:.8f}' if scale is not None else 'N/A'
        zp_str = f'{zp}' if zp is not None else 'N/A'
        f.write(f'{i:3d}. {layer:40s}  scale={scale_str:12s}  zero_point={zp_str:4s}\n')
    
    f.write(f'\n\nALL QUANTIZED LAYERS ({len(quantized_layers)} total):\n')
    f.write('='*80 + '\n')
    for i, (layer, scale, zp) in enumerate(quantized_layers, 1):
        scale_str = f'{scale:.8f}' if scale is not None else 'N/A'
        zp_str = f'{zp}' if zp is not None else 'N/A'
        f.write(f'{i:3d}. {layer:40s}  scale={scale_str:12s}  zero_point={zp_str:4s}\n')

print(f'Generated quantized_layers_simple_list.txt with {len(quantized_layers)} layers')
print(f'  - Direct layers (CoordAtt): {len(direct_layers)}')
print(f'  - Wrapped layers: {len(wrapped_layers)}')

