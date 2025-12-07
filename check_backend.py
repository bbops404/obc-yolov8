#!/usr/bin/env python3
"""Check the quantization backend used in a QAT or INT8 checkpoint."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'obc-yolov8' / 'ultralytics10.24'))

def check_backend(checkpoint_path):
    """Check the backend stored in a checkpoint."""
    print(f"Checking checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Check if backend is stored in checkpoint
        backend = checkpoint.get('backend', None)
        if backend:
            print(f"✓ Backend stored in checkpoint: {backend}")
        else:
            print("⚠️  No backend stored in checkpoint metadata")
        
        # Check current PyTorch backend setting
        current_backend = torch.backends.quantized.engine
        print(f"Current PyTorch backend: {current_backend}")
        
        # Check if it's QAT or INT8
        is_qat = checkpoint.get('qat', False)
        is_int8 = checkpoint.get('int8', False)
        
        if is_qat:
            print("Model type: QAT (Quantization-Aware Training)")
        elif is_int8:
            print("Model type: INT8 (Quantized)")
        else:
            print("Model type: Unknown (checking model structure...)")
            model = checkpoint.get('model', None)
            if model is not None:
                from torch.ao.quantization import FakeQuantize
                fakequant_count = sum(1 for m in model.modules() if isinstance(m, FakeQuantize))
                if fakequant_count > 0:
                    print(f"  Found {fakequant_count} FakeQuantize modules (likely QAT)")
                else:
                    print("  No FakeQuantize modules found (likely INT8 or FP32)")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION:")
        if backend == 'qnnpack':
            print("⚠️  This checkpoint uses 'qnnpack' backend (for ARM/M1)")
            print("   For x86 AWS, you should use 'fbgemm' backend")
            print("   Re-run training with: --backend fbgemm")
        elif backend == 'fbgemm':
            print("✓ This checkpoint uses 'fbgemm' backend (for x86)")
            print("   This is correct for AWS x86 instances")
        else:
            print(f"⚠️  Unknown backend: {backend}")
            print("   For x86 AWS, use 'fbgemm'")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_backend.py <checkpoint_path>")
        print("Example: python check_backend.py runs/detect/qat_fold4_qnnpack3/weights/last_qat.pt")
        sys.exit(1)
    
    checkpoint_path = Path(sys.argv[1])
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    check_backend(checkpoint_path)

