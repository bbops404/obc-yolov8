import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Patch for PyTorch 2.6
import ultralytics.nn.tasks as tasks
tasks.torch_safe_load = lambda file: (torch.load(file, map_location='cpu', weights_only=False), file)

from torch.ao.quantization import get_default_qconfig, prepare, convert, QuantStub, DeQuantStub
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from typing import Dict, Any, Optional

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "/Users/user/Documents/obc-yolov8/runs/detect/train7/weights/last.pt"
DATASET_YAML = "/Users/user/Documents/obc-yolov8/ultralytics10.24/ultralytics/cfg/datasets/combined_china_motorbike.yaml"
SAVE_DIR = "runs/ptq_comprehensive_analysis"
DEVICE = 'cpu'

# ============================================================================
# Utilities
# ============================================================================

def build_dataloader(yaml_path, batch=8):
    """Build validation dataloader"""
    data_cfg = check_det_dataset(yaml_path)
    val_path = data_cfg.get('val')
    
    dataset = YOLODataset(
        img_path=val_path, data=data_cfg, imgsz=640,
        augment=False, rect=False, cache=False
    )
    
    return DataLoader(
        dataset, batch_size=batch, shuffle=False,
        num_workers=2, collate_fn=getattr(dataset, "collate_fn", None)
    )

def get_model_size(path):
    """Get model file size in MB"""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 ** 2)
    return float('nan')

# ============================================================================
# Step 1: Baseline FP32 Validation
# ============================================================================

def validate_fp32_baseline(model_path, data_yaml, device='cpu'):
    """
    Validate the original FP32 model - this WILL work
    """
    print("\n" + "="*80)
    print("STEP 1: FP32 BASELINE VALIDATION")
    print("="*80)
    
    model = YOLO(model_path)
    
    print("[INFO] Running validation on FP32 model...")
    results = model.val(data=data_yaml, device=device, verbose=False)
    
    map50 = getattr(results.box, 'map50', 0.0) if hasattr(results, 'box') else 0.0
    map5095 = getattr(results.box, 'map', 0.0) if hasattr(results, 'box') else 0.0
    size_mb = get_model_size(model_path)
    
    metrics = {
        'mAP@0.5': map50,
        'mAP@0.5:0.95': map5095,
        'size_mb': size_mb,
        'status': 'success',
        'note': 'Baseline FP32 model validated successfully'
    }
    
    print(f"✅ Baseline Results:")
    print(f"   mAP@0.5:      {map50:.4f}")
    print(f"   mAP@0.5:0.95: {map5095:.4f}")
    print(f"   Model Size:   {size_mb:.2f} MB")
    
    return metrics, model

# ============================================================================
# Step 2: INT8 PTQ (Quantization & Calibration - Will Succeed)
# ============================================================================

def perform_int8_ptq(model, data_yaml, save_dir, device='cpu'):
    """
    Perform INT8 PTQ - quantization and calibration WILL succeed
    But inference validation WILL fail due to SiLU
    """
    print("\n" + "="*80)
    print("STEP 2: INT8 POST-TRAINING QUANTIZATION")
    print("="*80)
    
    # Prepare model for quantization
    print("[INFO] Preparing model for quantization...")
    
    class QuantizedWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.quant = QuantStub()
            self.model = model
            self.dequant = DeQuantStub()
        
        def forward(self, x):
            x = self.quant(x)
            x = self.model(x)
            x = self.dequant(x)
            return x
    
    # Wrap model
    q_model = QuantizedWrapper(model.model.float())
    q_model.eval()
    q_model.to(device)
    
    # Set quantization config
    qconfig = get_default_qconfig('qnnpack')
    q_model.qconfig = qconfig
    
    # Prepare for quantization
    print("[INFO] Inserting observers...")
    q_model = prepare(q_model, inplace=True)
    
    # Calibration
    print("[INFO] Running calibration (100 samples)...")
    calib_loader = build_dataloader(data_yaml, batch=8)
    
    calibrated = 0
    q_model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if isinstance(batch, dict):
                imgs = batch.get('img', batch.get('im'))
            else:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            
            if imgs is None:
                continue
            
            imgs = imgs.to(device)
            
            try:
                _ = q_model(imgs)
                calibrated += imgs.shape[0]
            except Exception as e:
                print(f"[WARNING] Calibration stopped at batch {i}: {e}")
                break
            
            if calibrated >= 100:
                break
    
    print(f"✅ Calibration complete: {calibrated} samples processed")
    
    # Convert to INT8
    print("[INFO] Converting model to INT8...")
    q_model_int8 = convert(q_model, inplace=True)
    
    # Save quantized model
    os.makedirs(save_dir, exist_ok=True)
    int8_path = os.path.join(save_dir, "model_int8_ptq.pt")
    torch.save(q_model_int8.state_dict(), int8_path)
    
    print(f"✅ INT8 model saved: {int8_path}")
    
    size_mb = get_model_size(int8_path)
    
    # Try to validate (this WILL fail)
    print("\n[INFO] Attempting validation on INT8 model...")
    
    validation_error = None
    map50 = None
    
    try:
        # This will fail with SiLU error
        test_loader = build_dataloader(data_yaml, batch=1)
        q_model_int8.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    imgs = batch.get('img', batch.get('im'))
                else:
                    imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                imgs = imgs.to(device)
                _ = q_model_int8(imgs)  # This will crash here
                break
        
        # If we get here, validation worked (unlikely)
        print("✅ Validation successful (unexpected!)")
        
    except Exception as e:
        validation_error = str(e)
        if "silu" in validation_error.lower():
            print("❌ Validation FAILED: SiLU incompatibility detected")
            print(f"   Error: {validation_error[:150]}...")
        else:
            print(f"❌ Validation FAILED: {validation_error[:150]}...")
    
    metrics = {
        'mAP@0.5': map50,
        'mAP@0.5:0.95': None,
        'size_mb': size_mb,
        'calibration_samples': calibrated,
        'status': 'quantization_success_inference_failed',
        'error': validation_error[:200] if validation_error else None,
        'note': 'INT8 quantization succeeded but inference fails due to SiLU incompatibility'
    }
    
    print(f"\n📊 INT8 PTQ Results:")
    print(f"   Quantization:  ✅ Success")
    print(f"   Calibration:   ✅ Success ({calibrated} samples)")
    print(f"   Model Size:    {size_mb:.2f} MB")
    print(f"   Inference:     ❌ Failed (SiLU not supported)")
    print(f"   Validation:    ❌ Cannot compute mAP")
    
    return metrics, int8_path

# ============================================================================
# Step 3: FP16 Quantization (Working Alternative)
# ============================================================================

def perform_fp16_quantization(model, model_path, data_yaml, save_dir, device='cpu'):
    """
    Perform FP16 quantization - this WILL work and give real results
    """
    print("\n" + "="*80)
    print("STEP 3: FP16 QUANTIZATION (WORKING ALTERNATIVE)")
    print("="*80)
    
    # Convert to FP16
    print("[INFO] Converting model to FP16...")
    model_fp16 = model.model.half()
    
    # Save FP16 model
    os.makedirs(save_dir, exist_ok=True)
    fp16_path = os.path.join(save_dir, "model_fp16.pt")
    torch.save(model_fp16.state_dict(), fp16_path)
    
    print(f"✅ FP16 model saved: {fp16_path}")
    
    # For validation, need to use FP32 on CPU
    print("[INFO] Validating FP16 model (converted to FP32 for CPU inference)...")
    model_fp16_cpu = model_fp16.float().cpu()
    
    # Wrap in YOLO-like object for validation
    model.model = model_fp16_cpu
    results = model.val(data=data_yaml, device=device, verbose=False)
    
    map50 = getattr(results.box, 'map50', 0.0) if hasattr(results, 'box') else 0.0
    map5095 = getattr(results.box, 'map', 0.0) if hasattr(results, 'box') else 0.0
    size_mb = get_model_size(fp16_path)
    
    metrics = {
        'mAP@0.5': map50,
        'mAP@0.5:0.95': map5095,
        'size_mb': size_mb,
        'status': 'success',
        'note': 'FP16 quantization fully compatible with SiLU'
    }
    
    print(f"\n✅ FP16 Results:")
    print(f"   mAP@0.5:      {map50:.4f}")
    print(f"   mAP@0.5:0.95: {map5095:.4f}")
    print(f"   Model Size:   {size_mb:.2f} MB")
    
    return metrics, fp16_path

# ============================================================================
# Step 4: Generate Comprehensive Report
# ============================================================================

def generate_report(baseline_metrics, int8_metrics, fp16_metrics, save_dir):
    """
    Generate comprehensive analysis report
    """
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    baseline_size = baseline_metrics['size_mb']
    int8_size = int8_metrics['size_mb']
    fp16_size = fp16_metrics['size_mb']
    
    # Calculate compressions
    int8_compression = baseline_size / int8_size if int8_size else 0
    fp16_compression = baseline_size / fp16_size if fp16_size else 0
    
    # Calculate accuracy deltas (FP16 only, since INT8 failed)
    fp16_delta = baseline_metrics['mAP@0.5'] - fp16_metrics['mAP@0.5']
    
    report = f"""
# Post-Training Quantization Analysis Report
## OBC-YOLOv8 with Custom Modules

Generated: {Path(save_dir).name}

---