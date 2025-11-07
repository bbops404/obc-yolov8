"""
Training script with proper QAT calibration settings
Ensures FakeQuantize observers collect statistics during training
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'obc-yolov8' / 'ultralytics10.24'))

from train_qat import train_qat

if __name__ == '__main__':
    # Configuration for proper QAT calibration
    config = {
        'model_cfg': "obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml",
        'data_cfg': "obc-yolov8/ultralytics10.24/ultralytics/cfg/datasets/combined_china_motorbike.yaml",
        'epochs': 20,  # quick QAT testing
        'imgsz': 640,
        'device': 0,  # GPU 0, or use 'cpu' for CPU
        'backend': 'fbgemm',  # x86 backend
        'save_qat_checkpoint': True,
        'convert_to_int8': True,
        'use_manual_quantization': False  # Use automatic quantization with fusion
    }
    
    print("=" * 80)
    print("QAT Training with Proper Calibration")
    print("=" * 80)
    print("This will:")
    print("  1. Fuse Conv+BN+Activation before QAT")
    print("  2. Enable FakeQuantize observers to collect statistics")
    print("  3. Train with quantization simulation")
    print("  4. Convert to INT8 after training")
    print("=" * 80)
    
    results = train_qat(**config)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Check calibration: python debug_calibration.py")
    print("  2. Evaluate INT8: python evaluate_int8.py --int8 runs/detect/train/weights/best_int8.pt")
    print("=" * 80)

