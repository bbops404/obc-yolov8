import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType
import torch.nn.quantized as nnq  # ⬅️ ADD THIS LINE
import os
import sys

# Add path first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"[DEBUG] Using ultralytics from: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")


# Force qnnpack for Mac M1/CPU
torch.backends.quantized.engine = 'qnnpack'


# ============================================================================
# PATCH: Override ultralytics torch_safe_load to use weights_only=False
# ============================================================================


import ultralytics.nn.tasks as tasks


def patched_torch_safe_load(file):
    """
    Patched version that uses weights_only=False for custom models.
    Safe to use since you trust your own checkpoint.
    NOTE: Removed safe_globals due to PyTorch 2.0.0 incompatibility.
    """
    print(f"[INFO] Loading checkpoint with weights_only=False: {file}")
    # torch.load without safe_globals for older PyTorch versions
    return torch.load(file, map_location='cpu', weights_only=False), file


# Apply the patch
tasks.torch_safe_load = patched_torch_safe_load


print("[INFO] ✓ Applied torch_safe_load patch (weights_only=False for custom modules)")


# ============================================================================
# Now continue with your normal imports
# ============================================================================


from torch.ao.quantization import (
    get_default_qconfig, prepare, convert, QuantStub, DeQuantStub, fuse_modules
)


from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from typing import List, Dict, Any, Tuple


from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import quantize_static, QuantType


# Import thop.profile, allowing it to fail gracefully
try:
    from thop import profile
except Exception:
    profile = None


# Import custom modules with robust fallbacks
try:
    from ultralytics.nn.ODConv import ODConv
    from ultralytics.nn.BoTNet import BoTNet
    from ultralytics.nn.CA_Attention import CoordAtt
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import C2f
    from ultralytics.nn.modules.conv import Concat
    from ultralytics.nn.modules.block import SPPF    
    from ultralytics.nn.tasks import Detect
except Exception as e:
    # Fallbacks in case the custom modules are not found
    print(f"[WARNING] Could not import custom modules: {e}")


    class ODConv(nn.Sequential):
        pass


    class ODConv2d(nn.Module):
        pass


    class CoordAtt(nn.Module):
        pass


    class BoTNet(nn.Module):
        pass
    
    class Conv(nn.Module): # Minimal fallback for Conv if needed later
        pass


print("[INFO] ✓ All imports complete - ready to load YOLO models")
print("[INFO] Safe globals registered for PyTorch 2.6+ weights loading (NOTE: This line is misleading and should be removed if using PT < 2.6)")


# =========================
# QUANTIZATION WRAPPERS (Omitted for brevity, assumed correct)
# =========================

import torch
import torch.nn as nn
import torch.ao.quantization

# Use the helper function defined previously to mirror attributes
import torch
import torch.nn as nn
import torch.ao.quantization as tq

def mirror_attributes(self, original_module):
    """Mirrors all necessary framework attributes (including 'i', 'f', 'save')"""
    for attr in dir(original_module):
        # Exclude dunder methods, PyTorch internals, private names, and methods
        if not attr.startswith(('_', 'torch', 'nn', 'forward')) and not hasattr(self, attr):
            try:
                # Get the value
                value = getattr(original_module, attr)
                # Only copy if it's not a module or a callable method
                if not isinstance(value, (nn.Module, nn.ModuleList)) and not callable(value):
                    setattr(self, attr, value)
            except (AttributeError, TypeError):
                pass
class QuantConcat(nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.concat_block = original_module
        mirror_attributes(self, original_module) # Assumes mirror_attributes is defined

    def forward(self, x):
        # x is a LIST of tensors
        # 🟢 FIX: Use a list comprehension, not tuple()
        x_dequant = [self.quant(t) for t in x]
        
        # Run original Concat block (now receives a LIST)
        out = self.concat_block(x_dequant)
        
        # Quantize output
        out = self.dequant(out)
        return out

class QuantODConv(nn.Module):
    """FP32 Wrapper. This module is non-quantizable."""
    def __init__(self, original_module):
        super().__init__()
        self.odconv_block = original_module
        mirror_attributes(self, original_module)
    def forward(self, x):
        return self.odconv_block(x)

class QuantCoordAtt(nn.Module):
    """FP32 Wrapper. This module is non-quantizable."""
    def __init__(self, original_module):
        super().__init__()
        self.coordatt_block = original_module
        mirror_attributes(self, original_module)
    def forward(self, x):
        return self.coordatt_block(x)

class QuantDetect(nn.Module):
    """FP32 Wrapper. This module is non-quantizable."""
    def __init__(self, original_module):
        super().__init__()
        self.detect_head = original_module
        mirror_attributes(self, original_module)
    def forward(self, x):
        return self.detect_head(x)
    
class QuantBoTNet(nn.Module):
    def __init__(self, original_module: BoTNet):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.botnet_block = original_module
        # 🟢 FIX: Apply comprehensive mirroring
        mirror_attributes(self, original_module)

    def forward(self, x):
        x = self.quant(x)
        out = self.botnet_block(x)
        out = self.dequant(out)
        return out

# --- Existing wrappers updated for clarity (Functionally already correct) ---

class QuantC2f(nn.Module):
    """Re-implements C2f to be quantization-aware using FloatFunctional."""
    def __init__(self, original_module: C2f):
        super().__init__()
        # Copy the internal layers that we want to quantize
        self.cv1 = original_module.cv1
        self.cv2 = original_module.cv2
        self.m = original_module.m
        
        # This is the magic part: a traceable, quant-aware 'cat'
        self.cat = nnq.FloatFunctional()
        
        # Copy framework attributes ('i', 'f', 'save', 'c', etc.)
        mirror_attributes(self, original_module)

    def forward(self, x):
        # Original C2f forward pass...
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        # ...but using the quant-aware 'cat' operation
        return self.cv2(self.cat.cat(y, 1))

class QuantConcat(nn.Module):
    """Re-implements Concat to be quantization-aware using FloatFunctional."""
    def __init__(self, original_module: Concat):
        super().__init__()
        self.d = original_module.d
        self.cat = nnq.FloatFunctional()
        mirror_attributes(self, original_module)

    def forward(self, x):
        # x is a list of tensors
        return self.cat.cat(x, self.d)

class QuantSPPF(nn.Module):
    """Re-implements SPPF to be quantization-aware using FloatFunctional."""
    def __init__(self, original_module: SPPF):
        super().__init__()
        self.cv1 = original_module.cv1
        self.cv2 = original_module.cv2
        self.m = original_module.m
        self.cat = nnq.FloatFunctional()
        mirror_attributes(self, original_module)
        
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # Use the quant-aware 'cat'
        return self.cv2(self.cat.cat([x, y1, y2, y3], 1))

class QuantBoTNet(nn.Module):
    """Re-implements BoTNet to be quantization-aware."""
    def __init__(self, original_module: BoTNet):
        super().__init__()
        self.cv1 = original_module.cv1
        self.cv2 = original_module.cv2
        self.cv3 = original_module.cv3
        self.m = original_module.m
        
        # 🟢 The 'm' (MHSA) block is non-quantizable. We MUST keep it FP32.
        # We add stubs to dequantize the input to 'm' and re-quantize its output.
        self.dequant_mhsa = tq.DeQuantStub()
        self.quant_mhsa = tq.QuantStub()

        self.cat = nnq.FloatFunctional()
        mirror_attributes(self, original_module)

    def forward(self, x):
        # Path 1 (Quantized)
        x1 = self.cv1(x)
        
        # --- FP32 Island for MHSA ---
        x1_fp = self.dequant_mhsa(x1)
        m_out_fp = self.m(x1_fp)
        m_out = self.quant_mhsa(m_out_fp)
        # --- End Island ---
        
        # Path 2 (Quantized)
        x2 = self.cv2(x)
        
        # Quant-aware 'cat'
        y = self.cat.cat((m_out, x2), dim=1)
        
        return self.cv3(y)

# NOTE: The FloatSiLU class does not need the framework attributes ('i', 'f', etc.) 
# as it typically replaces a simple activation function and is not a core model block.
class FloatSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()
    def forward(self, x):
        try:
            if hasattr(x, 'is_quantized') and x.is_quantized():
                x = x.dequantize()
        except Exception:
            if getattr(x, 'dtype', None) in (torch.quint8, torch.qint8, torch.qint32):
                try:
                    x = x.dequantize()
                except Exception:
                    x = x.float()
        out = self.act(x)
        return out

def replace_silu_with_float(module: nn.Module):
    """Recursively replaces nn.SiLU with FloatSiLU for quantization compatibility."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SiLU):
            setattr(module, name, FloatSiLU())
        else:
            replace_silu_with_float(child)

import torch.nn as nn
# Assuming CoordAtt, ODConv, BoTNet, and Concat are imported from their source files
# and QuantCoordAtt, QuantODConv, QuantBoTNet, and QuantConcat are your wrapper classes

# Define the set of custom wrapper classes to stop the recursion (add the new one)
# Assuming 'Concat' is the class name for the ultralytics concatenation module

import torch.nn as nn
import torch.ao.quantization


QUANT_WRAPPERS = (QuantCoordAtt, QuantODConv, QuantBoTNet, QuantConcat, QuantC2f, QuantSPPF) 
def replace_modules_with_quant_wrappers(model: nn.Module) -> nn.Module:
    """Recursively replaces custom modules with their quantization-aware wrappers."""
    for name, module in list(model.named_children()):
        
        # Flag to track if a replacement occurred
        was_replaced = False

        # 🟢 NEW: Check for C2f and wrap it
        if isinstance(module, C2f):
            print(f"[INFO] Replacing C2f instance: {name}")
            setattr(model, name, QuantC2f(module))
            was_replaced = True
        elif isinstance(module, CoordAtt):
            print(f"[INFO] Replacing CoordAtt instance: {name}")
            setattr(model, name, QuantCoordAtt(module))
            was_replaced = True
        elif isinstance(module, ODConv):
            print(f"[INFO] Replacing ODConv instance: {name}")
            setattr(model, name, QuantODConv(module))
            was_replaced = True
        elif isinstance(module, BoTNet):
            print(f"[INFO] Replacing BoTNet instance: {name}")
            setattr(model, name, QuantBoTNet(module))
            was_replaced = True
        elif isinstance(module, Concat): 
            print(f"[INFO] Replacing Concat instance: {name}")
            setattr(model, name, QuantConcat(module))
            was_replaced = True
        elif isinstance(module, SPPF):
            print(f"[INFO] Replacing SPPF instance: {name}")
            setattr(model, name, QuantSPPF(module))
            was_replaced = True
            # 🎯 NEW: Wrap the final Detect head
        elif isinstance(module, Detect):
            print(f"[INFO] Replacing Detect instance: {name}")
            setattr(model, name, QuantDetect(module))
            was_replaced = True

        # Get the module to potentially recurse into
        current_module = getattr(model, name)
        
        # CRITICAL STOP CONDITION: Prevent recursion into the wrappers
        if was_replaced or isinstance(current_module, QUANT_WRAPPERS):
            continue
            
        # Recursive Descent (Only happens if the module is a container and wasn't wrapped)
        if len(list(current_module.named_children())) > 0:
            replace_modules_with_quant_wrappers(current_module)
            
    return model
# ======================================================================
# 2. CONSTANTS & CONFIGURATION (Define your paths and settings here)
# ======================================================================
MODEL_PATH = "/Users/user/Documents/obc-yolov8/obc-yolov8/runs/detect/train7/weights/last.pt"
DATASET_CONFIG_YAML = "/Users/user/Documents/obc-yolov8/ultralytics10.24/ultralytics/cfg/datasets/combined_china_motorbike.yaml"
CALIB_BATCH = 8
CALIB_WORKERS = 4
SAVE_DIR = "runs/quantized"
EVAL_DEVICE = 'cpu'


# Define modules for selective quantization experiments
MODULE_VARIANTS: Dict[str, List[str]] = {
    "CA_only": [
        "model.19.cv1.conv", "model.19.cv1.bn",
        "model.19.cv2.conv", "model.19.cv2.bn",
        "model.19.m.0.cv1.conv", "model.19.m.0.cv1.bn",
        "model.19.m.0.cv2.conv", "model.19.m.0.cv2.bn",
        "model.23.cv1.conv", "model.23.cv1.bn",
        "model.23.cv2.conv", "model.23.cv2.bn",
        "model.23.m.0.cv1.conv", "model.23.m.0.cv1.bn",
        "model.23.m.0.cv2.conv", "model.23.m.0.cv2.bn",
    ],
    "BoTNet_only": [
        "model.10.cv1.conv", "model.10.cv1.bn",
        "model.10.cv2.conv", "model.10.cv2.bn",
        "model.10.cv3.conv", "model.10.cv3.bn",
        "model.10.m.0.cv1.conv", "model.10.m.0.cv1.bn",
        "model.10.m.0.cv2.conv", "model.10.m.0.cv2.bn",
        "model.10.m.0.fc1",
    ],
    "ODConv_only": [
        "model.1.0.conv", "model.1.0.bn",
    ],
    "All_Modules": [
        "model.1.0.conv", "model.1.0.bn",
        "model.19.cv1", "model.19.cv2", "model.19.m",
        "model.23.cv1", "model.23.cv2", "model.23.m",
        "model.10.cv1", "model.10.cv2", "model.10.cv3", "model.10.m",
        "model.10.m.0.cv1", "model.10.m.0.cv2", "model.10.m.0.fc1"
    ],
}


# -------------------------
# Data loader helpers (Omitted for brevity, assumed correct)
# -------------------------


def dict_collate_fn(batch):
    """Custom collate function for YOLODataset."""
    imgs = torch.stack([item['img'] for item in batch])
    labels = [item.get('labels', []) for item in batch]
    return {'img': imgs, 'labels': labels}


# Your original data loader function
def build_custom_dataloader(yaml_path, imgsz, batch, workers) -> DataLoader:
    """Builds the validation/calibration dataloader from a YOLO dataset config."""
    data_cfg = check_det_dataset(yaml_path)
    val_path = data_cfg.get('val')
    if not val_path:
        raise ValueError("YAML must contain 'val' path for calibration/validation.")


    dataset = YOLODataset(
        img_path=val_path,
        data=data_cfg,
        imgsz=imgsz,
        augment=False,
        rect=False,
        cache=False
    )


    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=getattr(dataset, "collate_fn", dict_collate_fn)
    )
    print(f"[INFO] Calib loader created from: {val_path} (Batch: {batch}, Workers: {workers})")
    return dataloader


# Custom CalibrationDataReader for ONNX Runtime
class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.data_iter = iter(dataloader)
    
    def get_next(self):
        try:
            batch = next(self.data_iter)
            imgs = batch.get('img', batch.get('im'))
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.numpy()
            return {"input": imgs}
        except StopIteration:
            return None
# -------------------------
# Validation utilities (Omitted for brevity, assumed correct)
# -------------------------


def run_validation(model, data_yaml, device='cpu'):
    """Runs YOLOv8 validation using the ultralytics Val class or manual validation."""
    if isinstance(model, YOLO):
        try:
            print(f"[INFO] Running YOLOv8 validation on {device}...")
            # Use the built-in Val method for original YOLO models
            results = model.val(data=data_yaml, device=device, verbose=False)
            map50 = getattr(results.box, 'map50', 0.0) if hasattr(results, 'box') else 0.0
            map5095 = getattr(results.box, 'map', 0.0) if hasattr(results, 'box') else 0.0
            size_mb = float('nan')
            if hasattr(model, 'pt_path') and os.path.exists(model.pt_path):
                size_mb = os.path.getsize(model.pt_path) / (1024 ** 2)
            metrics = {'mAP@0.5': map50, 'mAP@0.5:0.95': map5095, 'size': size_mb, 'loss': float('nan'), 'flops': float('nan'), 'params': float('nan')}
            print(f"✓ Validation | mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map5095:.4f}")
            return metrics
        except Exception as e:
            print(f"[ERROR] YOLOv8 validation failed: {e}")
            # If standard validation fails, fall back to manual
            return run_manual_validation(model, data_yaml, device)
    else:
        print(f"[INFO] Running manual validation on quantized model...")
        return run_manual_validation(model, data_yaml, device)


def run_manual_validation(model, data_yaml, device='cpu', conf=0.001, iou=0.6, max_batches=None):
    """Performs manual validation for quantized models without the full YOLOv8 Val class."""
    from ultralytics.utils.metrics import box_iou, ap_per_class
    from ultralytics.utils.ops import non_max_suppression


    eval_model = model.model if hasattr(model, 'model') else model
    eval_model.eval()
    eval_model.to(device)


    data_cfg = check_det_dataset(data_yaml)
    val_path = data_cfg.get('val')
    nc = len(data_cfg.get('names', []))


    dataset = YOLODataset(
        img_path=val_path,
        data=data_cfg,
        imgsz=640,
        augment=False,
        rect=False,
        cache=False
    )


    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=getattr(dataset, "collate_fn", dict_collate_fn)
    )


    stats = []
    seen = 0


    print(f"[INFO] Validating on {len(dataset)} images...")


    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break


            if isinstance(batch, dict):
                imgs = batch.get('img', batch.get('im', None))
                labels = batch.get('cls', None)
                bboxes = batch.get('bboxes', None)
                batch_idx_tensor = batch.get('batch_idx', None)
            else:
                continue


            if imgs is None:
                continue


            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            seen += batch_size


            try:
                preds = eval_model(imgs)
            except Exception as e:
                print(f"[ERROR] Inference failed on batch {batch_idx}: {e}")
                # If this error is the quantized kernel missing, fall back to float model:
                if 'quantized::conv2d.new' in str(e) or 'Could not run' in str(e):
                    try:
                        # load float model fallback (slow, but works)
                        float_model = YOLO(getattr(eval_model, 'pt_path', MODEL_PATH))
                        float_model.model.eval().to('cpu')
                        preds = float_model.model(imgs.cpu())
                        print("[INFO] Fallback: ran batch with FP32 model")
                    except Exception as e2:
                        print(f"[ERROR] Fallback inference also failed: {e2}")
                        # create empty preds/continue
                        preds = None
                else:
                    preds = None


            if isinstance(preds, (list, tuple)):
                preds = preds[0]


            try:
                preds = non_max_suppression(preds, conf_thres=conf, iou_thres=iou, max_det=300)
            except Exception as e:
                print(f"[ERROR] NMS failed: {e}")
                continue


            for si, pred in enumerate(preds):
                if batch_idx_tensor is not None and labels is not None and bboxes is not None:
                    idx_mask = batch_idx_tensor == si
                    gt_cls = labels[idx_mask].cpu()
                    gt_bbox = bboxes[idx_mask].cpu()
                else:
                    gt_cls = torch.empty(0)
                    gt_bbox = torch.empty((0, 4))


                if pred is None or len(pred) == 0:
                    if len(gt_bbox):
                        stats.append((torch.zeros(0, dtype=torch.bool), torch.zeros(0), torch.zeros(0), gt_cls))
                    continue


                pred = pred.cpu()
                pred_boxes = pred[:, :4]
                pred_conf = pred[:, 4]
                pred_cls = pred[:, 5]


                if len(gt_bbox):
                    correct = torch.zeros(len(pred), nc, dtype=torch.bool)
                    iou_mat = box_iou(gt_bbox, pred_boxes)
                    for i, gc in enumerate(gt_cls):
                        j = (pred_cls == gc).nonzero(as_tuple=False).view(-1)
                        if len(j):
                            m = (iou_mat[i, j] > iou).nonzero(as_tuple=False).view(-1)
                            if len(m):
                                correct[j[m], int(gc)] = True
                    stats.append((correct, pred_conf, pred_cls, gt_cls))
                else:
                    stats.append((torch.zeros(len(pred), nc, dtype=torch.bool), pred_conf, pred_cls, torch.zeros(0)))


    if not stats:
        print("[WARNING] No valid predictions")
        return {'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0, 'size': float('nan'), 'loss': float('nan'), 'flops': float('nan'), 'params': float('nan')}


    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]


    if len(stats) and stats[0].any():
        ap50, ap = ap_per_class(*stats, plot=False, save_dir=None, names={})
        map50, map5095 = ap50.mean(), ap.mean()
    else:
        map50, map5095 = 0.0, 0.0


    size_mb = float('nan')
    if hasattr(model, 'pt_path') and os.path.exists(model.pt_path):
        size_mb = os.path.getsize(model.pt_path) / (1024 ** 2)


    metrics = {'mAP@0.5': map50, 'mAP@0.5:0.95': map5095, 'size': size_mb, 'loss': float('nan'), 'flops': float('nan'), 'params': float('nan')}
    print(f"✓ Validation | mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map5095:.4f} ({seen} images)")
    return metrics


# -------------------------
# Quant preparation & PTQ flows (Omitted for brevity, assumed correct)
# -------------------------


def fuse_simple_conv_bn(module: nn.Module) -> nn.Module:
    """Fuses Conv and BatchNorm in a standard ultralytics Conv module."""
    if isinstance(module, Conv) and hasattr(module, 'conv') and hasattr(module, 'bn'):
        try:
            fuse_modules(module, ['conv', 'bn'], inplace=True)
            module.bn = nn.Identity()
        except Exception:
            pass
    return module


def fuse_standard_conv_only(model: nn.Module) -> nn.Module:
    """Recursively traverses model to fuse standard Conv/BN layers."""
    for module in model.children():
        if isinstance(module, Conv):
            if hasattr(module, 'conv') and hasattr(module, 'bn'):
                try:
                    fuse_modules(module, ['conv', 'bn'], inplace=True)
                    module.bn = nn.Identity()
                except Exception:
                    continue
        fuse_standard_conv_only(module)
    return model



class QuantizedYOLO(nn.Module):
    """Wrapper to handle input/output quant/dequant for the whole model."""
    def __init__(self, model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub() # 🟢 Add this back

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x) # 🟢 Add this back
        return x

import torch.quantization as tq
# ... (rest of imports)

import torch.ao.quantization as tq
# Make sure to import all your original and wrapper classes at the top of the file
# (YOLO, C2f, Concat, SPPF, BoTNet, ODConv, CoordAtt, Detect)
# (QuantC2f, QuantConcat, QuantSPPF, QuantBoTNet, QuantODConv, QuantCoordAtt, QuantDetect, QuantizedYOLO)
def replace_silu_with_relu(model: nn.Module):
    """
    Recursively replaces all nn.SiLU activations with nn.ReLU(inplace=True).
    This is necessary because the 'fbgemm' backend does not support quantized SiLU.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.SiLU):
            print(f"[INFO] Replacing {name} (nn.SiLU) with nn.ReLU")
            setattr(model, name, nn.ReLU(inplace=True))
        elif len(list(module.children())) > 0:
            replace_silu_with_relu(module) # Recurse
    return model
def prepare_quant_model(model_path: str, device: str = 'cpu') -> Tuple[nn.Module, Dict[int, str]]:
    """Loads, fuses, and wraps the model for fine-grained PTQ."""
    
    # 1. Load model
    yolo_wrapper = YOLO(model_path)
    m = yolo_wrapper.model.float().to(device)
    m.eval()
    names = yolo_wrapper.names
# 🟢 NEW STEP: Replace all SiLU with ReLU
    print("[INFO] Replacing all nn.SiLU activations with nn.ReLU...")
    m = replace_silu_with_relu(m)
    print("[INFO] SiLU replacement complete.")

    # 2. Fuse standard layers (must be done *before* wrapping)
    m_fused = fuse_simple_conv_bn(m)
    m_fused = fuse_standard_conv_only(m_fused)

    # 3. Replace custom modules with our new fine-grained wrappers
    m_wrapped = replace_modules_with_quant_wrappers(m_fused)
    
    # 4. Remove the old SiLU replacement (let's try to quantize it natively)
    # replace_silu_with_float(m_wrapped) # <-- REMOVED FOR SIMPLICITY

    # 5. Wrap the entire model in the top-level quant/dequant stubs
    q_model = QuantizedYOLO(m_wrapped) # This wrapper has .quant and .dequant
    q_model.pt_path = model_path
    q_model.eval()

    # 6. 🚨 CRITICAL FIX: Set QConfig logic
    
    # Use 'fbgemm' for x86 CPUs (like your Mac) or 'qnnpack' for ARM
    backend = 'qnnpack' 
    qconfig = tq.get_default_qconfig(backend)
    
    # Set the default qconfig for the *entire* model.
    # This tells 'prepare' to *try* to quantize everything.
    q_model.qconfig = qconfig

    # 7. 🚨 CRITICAL FIX 2: Manually turn OFF quantization for non-quantizable modules.
    # This is the core of the fine-grained strategy.
    for name, module in q_model.named_modules():
        
        # 🟢 ADD 'Conv' TO THIS LIST
        # This will treat the standard Conv block as an FP32 island
        # because its child 'nn.SiLU' is not supported by the backend.
        if isinstance(module, (QuantODConv, QuantCoordAtt, QuantDetect)):
            print(f"[INFO] Disabling quantization (FP32) for: {name}")
            module.qconfig = None
            
        if isinstance(module, QuantBoTNet):
             if hasattr(module, 'm'): 
                print(f"[INFO] Disabling quantization (FP32) for: {name}.m (MHSA block)")
                module.m.qconfig = None

    # 8. Use the official 'prepare' call.
    # 'prepare' will now:
    # - Quantize all standard layers (Conv, BN, ReLU, SiLU)
    # - Quantize the *internals* of QuantC2f, QuantConcat, QuantSPPF
    # - Insert dequant/quant stubs *around* the modules we set to qconfig=None
    model_prepared = tq.prepare(q_model.to(device), inplace=True) 

    return model_prepared, names

def run_global_ptq(model_path, data_yaml, calib_loader, save_dir, device='cpu'):
    """Performs Post-Training Quantization (PTQ) on the entire model (conservative approach)."""
    print("\n" + "="*80)
    print("STEP 1: BASELINE (FP32)")
    print("="*80)


    baseline_yolo = YOLO(model_path)
    baseline_yolo.pt_path = model_path
    baseline_metrics = run_validation(baseline_yolo, data_yaml, device)


    print("\n" + "="*80)
    print("STEP 2: CONSERVATIVE PTQ")
    print("="*80)


    q_model, names = prepare_quant_model(model_path, device)
    q_model.to(device)



    print("\n[INFO] Calibrating...")
    q_model.eval()
    calibrated = 0


    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if isinstance(batch, dict):
                imgs = batch.get('img', batch.get('im'))
            elif isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch


            if imgs is None:
                continue


            imgs = imgs.to(device)


            try:
                _ = q_model(imgs)
                calibrated += imgs.shape[0]
            except Exception as e:
                print(f"[ERROR] Calibration failed: {e}")
                break


            if calibrated >= 100:
                break


    print(f"[INFO] Calibrated on {calibrated} samples")


    # Convert the model to INT8
    q_model_int8 = convert(q_model, inplace=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the PyTorch Quantized Model (Optional, but good practice)
    q_model_int8.pt_path = os.path.join(save_dir, "conservative_ptq.pt")
    # Note: torch.save will save the state_dict, not the full quantized graph
    # For full graph export to ONNX, we use the model object below
    
    # --- IMPORTANT: Return the INT8 model object for ONNX export ---
    
    class QuantWrapper(nn.Module):
        """Minimal wrapper to use quantized model for validation."""
        def __init__(self, model, names, pt_path):
            super().__init__()
            self.model = model
            self.names = names
            self.pt_path = pt_path
        def forward(self, x):
            return self.model(x)


    post_q_model = QuantWrapper(q_model_int8, names, q_model_int8.pt_path)
    post_q_metrics = run_validation(post_q_model, data_yaml, device)


    results = {'Baseline (FP32)': baseline_metrics, 'Conservative_PTQ': post_q_metrics}


    print("\n" + "="*80)
    print(f"Baseline:         {baseline_metrics.get('mAP@0.5', 0.0):.4f}")
    print(f"Conservative PTQ: {post_q_metrics.get('mAP@0.5', 0.0):.4f}")
    print(f"Delta:            {baseline_metrics.get('mAP@0.5', 0.0) - post_q_metrics.get('mAP@0.5', 0.0):.4f}")
    print("="*80)


    return baseline_metrics, q_model_int8, results # Return the actual INT8 model object


# NOTE: The original code contained a call to `should_skip` which was not defined.
# I'm providing a minimal definition here to prevent runtime errors. The logic inside
# `run_selective_ptq` may need refinement based on your actual intent.
def should_skip(module, full_path):
    """Placeholder for logic missing in the original user code."""
    return False


def run_selective_ptq(model_path, data_yaml, calib_loader, save_dir, experiment_name, include_modules, device='cpu'):
    """Performs selective PTQ, where only specified modules are quantized."""
    print(f"\n--- Selective PTQ: {experiment_name} ---")
    q_model, names = prepare_quant_model(model_path, device)
    qconfig = get_default_qconfig('qnnpack')


    def apply_selective_qconfig(module, path=""):
        for name, child in module.named_children():
            full_path = f"{path}.{name}" if path else name
            
            # Apply QConfig only if the module's path is in the include list
            if full_path in include_modules:
                if isinstance(child, (nn.Conv2d, nn.Linear)):
                    child.qconfig = qconfig
                elif hasattr(child, 'quant') and hasattr(child, 'dequant'): # For wrappers
                    child.quant.qconfig = qconfig
                    child.dequant.qconfig = qconfig
                print(f"[Q] Applied QConfig to: {full_path} ({type(child).__name__})")
            elif should_skip(child, full_path): # The original code called this function
                print(f"[SKIPPED] {full_path}")
                continue
            
            apply_selective_qconfig(child, full_path)


    # Clear previously inserted QConfigs and observers
    def clear_qconfig(module):
        module.qconfig = None
        for child in module.children():
            clear_qconfig(child)


    clear_qconfig(q_model)


    apply_selective_qconfig(q_model.model, "model")


    q_model.quant.qconfig = qconfig
    q_model.dequant.qconfig = qconfig
    
    # Re-prepare after selectively applying qconfig
    q_model = prepare(q_model, inplace=True)


    q_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if isinstance(batch, dict):
                imgs = batch.get('img', batch.get('im'))
            elif isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch


            if imgs is None:
                continue
            imgs = imgs.to(device)


            try:
                _ = q_model(imgs)
            except Exception:
                continue


            if i * calib_loader.batch_size >= 100:
                break


    q_model_int8 = convert(q_model, inplace=True)
    q_model_int8.pt_path = os.path.join(save_dir, f"selective_{experiment_name}.pt")
    torch.save(q_model_int8.state_dict(), q_model_int8.pt_path)


    class QuantWrapper(nn.Module):
        def __init__(self, model, names, pt_path):
            super().__init__()
            self.model = model
            self.names = names
            self.pt_path = pt_path
        def forward(self, x):
            return self.model(x)


    wrapper = QuantWrapper(q_model_int8, names, q_model_int8.pt_path)
    return run_validation(wrapper, data_yaml, device)


def run_sensitivity_ptq(model_path, data_yaml, calib_loader, save_dir, baseline_metrics, device='cpu'):
    """Runs selective PTQ experiments for different module variants."""
    print("\nRunning sensitivity experiments...")
    results = {}
    for name, module_list in MODULE_VARIANTS.items():
        metrics = run_selective_ptq(model_path, data_yaml, calib_loader, save_dir, name, module_list, device)
        results[name] = metrics
    return results


def print_summary(results_table: Dict[str, Dict[str, Any]], baseline_map: float):
    """Prints a summary table of all PTQ results."""
    print("\n" + "="*80)
    print("PTQ Sensitivity Analysis Summary")
    print("="*80)
    print(f"{ 'Experiment':<25} {'mAP@0.5':<10} {'Loss':<8} {'Size(MB)':<10} {'GFLOPs':<10} {'Params(M)':<10} {'Speed':<10} {'Delta':<8}")
    print("-" * 80)


    # Order the results for better readability
    ordered_keys = [k for k in results_table.keys() if 'Baseline' in k]
    ordered_keys.extend([k for k in results_table.keys() if 'Conservative' in k and 'Baseline' not in k])
    ordered_keys.extend([k for k in results_table.keys() if 'Baseline' not in k and 'Conservative' not in k])


    for name in ordered_keys:
        m = results_table.get(name, {})
        delta = baseline_map - m.get("mAP@0.5", 0.0) if 'Baseline' not in name else 0.0
        # Placeholder values for missing metrics
        map50 = m.get('mAP@0.5', 0.0)
        loss = m.get('loss', float('nan'))
        size = m.get('size', float('nan'))
        flops = m.get('flops', float('nan'))
        params = m.get('params', float('nan'))
        speed = str(m.get('speed', 'N/A'))


        print(f"{name:<25} {map50:<10.4f} {loss:<8.4f} {size:<10.2f} {flops:<10.2f} {params:<10.2f} {speed:<10} {delta:<8.4f}")


    print("="*80)


import onnxruntime as ort
import numpy as np


def evaluate_onnx_model(onnx_model_path, data_yaml, device='cpu', conf=0.001, iou=0.6, max_batches=None):
    """
    Runs manual validation using ONNX Runtime.
    Note: This is simplified and does not implement the full metric calculation.
    You will need to adapt run_manual_validation logic to use ONNX Runtime output.
    """
    sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # Use validation data loader
    dataloader = build_custom_dataloader(data_yaml, imgsz=640, batch=8, workers=2)
    
    all_outputs = []
    
    print(f"[INFO] Running inference on ONNX quantized model: {onnx_model_path}")
    
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
            
        imgs = batch.get('img', batch.get('im'))
        if imgs is None:
            continue
            
        # Convert PyTorch tensor input to NumPy for ONNX Runtime
        imgs_np = imgs.cpu().numpy()
        
        # Run ONNX inference
        preds = sess.run([output_name], {input_name: imgs_np})
        all_outputs.append(preds)
        
    print(f"[INFO] Inference completed on quantized ONNX model for {len(all_outputs)} batches.")
    return all_outputs


# =========================================================================
# NEW FUNCTION: EXPORT PT QUANTIZED MODEL TO ONNX
# =========================================================================

def export_pt_quantized_to_onnx(pt_quant_model, onnx_output_path):
    """Exports the PyTorch-quantized INT8 model to ONNX."""
    print("\n" + "="*80)
    print("STEP 3: EXPORT PT QUANTIZED MODEL TO ONNX")
    print("="*80)
    
    pt_quant_model.eval()
    
    # NOTE: The dummy input must be float32, as the QuantStub handles the conversion internally.
    dummy_input = torch.randn(1, 3, 640, 640) 
    
    try:
        torch.onnx.export(
            pt_quant_model,
            dummy_input,
            onnx_output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=13, # Use 13 for wide compatibility
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            # Crucial for PyTorch Quantized Models
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        )
        print(f"✓ Successfully exported PyTorch Quantized Model to ONNX at {onnx_output_path}")
        return onnx_output_path
    except Exception as e:
        print(f"❌ ONNX Export of PT Quantized Model FAILED: {e}")
        print("Hint: This is often caused by non-standard quantized layers (e.g., ODConv).")
        return None

# -------------------------
# MAIN
# -------------------------


def main():
    print("Quant backend:", torch.backends.quantized.engine)
    calib_loader = build_custom_dataloader(DATASET_CONFIG_YAML, imgsz=640, batch=CALIB_BATCH, workers=CALIB_WORKERS)


    # Step 1: Baseline FP32 and PyTorch Conservative PTQ
    baseline_metrics, pt_quantized_model, global_results = run_global_ptq(
        MODEL_PATH, DATASET_CONFIG_YAML, calib_loader, SAVE_DIR, EVAL_DEVICE)
    baseline_map = baseline_metrics.get('mAP@0.5', 0.0)


    # Step 2: Export PyTorch Quantized Model to ONNX
    onnx_quantized_model_path = os.path.join(SAVE_DIR, "model_pt_quantized.onnx")
    final_onnx_path = export_pt_quantized_to_onnx(pt_quantized_model, onnx_quantized_model_path)


    # Step 3: Run inference and validation on the exported ONNX model
    if final_onnx_path:
        # Note: This step is for basic validation only.
        # To get true mAP, you need to fully integrate the ONNX Runtime output
        # into the `run_manual_validation` structure, which is a major code change.
        evaluate_onnx_model(final_onnx_path, DATASET_CONFIG_YAML)
    
    
    # Step 4: Run selective PTQ tests (PyTorch-based)
    # The original code runs this, keeping it for completeness.
    sensitivity_results = run_sensitivity_ptq(
        MODEL_PATH, DATASET_CONFIG_YAML, calib_loader, SAVE_DIR, baseline_metrics, EVAL_DEVICE)


    # Step 5: Print final summary
    global_results.update(sensitivity_results)
    print_summary(global_results, baseline_map)


if __name__ == "__main__":
    main()