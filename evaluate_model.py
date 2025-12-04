"""
Evaluate YOLO model on validation/test dataset with full metrics.
Now exports FP32 model to inference-only format before evaluation.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add ultralytics to path
REPO_ROOT = Path(__file__).resolve().parent
ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics10.24"
sys.path.insert(0, str(ULTRALYTICS_ROOT))

from ultralytics import YOLO


def export_model(model, export_format: str, output_dir: Path) -> Path:
    """
    Export model to inference-only format.
    
    Args:
        model: YOLO model instance
        export_format: 'torchscript' or 'onnx'
        output_dir: Directory to save exported model
        
    Returns:
        Path to exported model
    """
    print(f"\n{'='*60}")
    print(f"Exporting model to {export_format.upper()} format...")
    print(f"{'='*60}")
    
    if export_format == 'torchscript':
        # Use torch.jit.script instead of trace for dynamic control flows
        # trace records a fixed forward pass and fails for dynamic shapes
        # script compiles the Python code itself, supporting dynamic control flows
        model.model.eval()
        
        # Get the model path for naming
        model_name = Path(model.ckpt_path).stem if hasattr(model, 'ckpt_path') else 'model'
        exported_path = output_dir / f"{model_name}.torchscript"
        
        print("Using torch.jit.script for dynamic control flow support...")
        scripted_model = torch.jit.script(model.model)
        scripted_model.save(str(exported_path))
        
        print(f"✓ Model exported to: {exported_path}")
        return exported_path
    
    else:
        # For ONNX, use YOLO's built-in export
        exported_path = model.export(
            format=export_format,
            imgsz=640,
            half=False,  # Keep FP32 precision
            simplify=True if export_format == 'onnx' else False,
            device='cpu',  # Force CPU for export
        )
        
        print(f"✓ Model exported to: {exported_path}")
        return Path(exported_path)


def get_model_size(model_path: Path) -> float:
    """Get model file size in MB."""
    return os.path.getsize(model_path) / (1024 * 1024)


def measure_latency_pytorch(model, imgsz: int = 640, warmup: int = 10, runs: int = 100) -> dict:
    """
    Measure inference latency for PyTorch model.
    
    Args:
        model: YOLO model
        imgsz: Input image size
        warmup: Number of warmup iterations
        runs: Number of measurement runs
        
    Returns:
        Dictionary with latency statistics
    """
    device = torch.device('cpu')  # Use CPU for consistent benchmarking
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    
    print(f"\nMeasuring PyTorch model latency (CPU, {runs} runs)...")
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.model(dummy_input)
    
    # Measure
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.model(dummy_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'fps': float(1000 / np.mean(latencies))
    }


def measure_latency_torchscript(model_path: Path, imgsz: int = 640, warmup: int = 10, runs: int = 100) -> dict:
    """
    Measure inference latency for TorchScript model.
    """
    device = torch.device('cpu')
    
    # Load TorchScript model
    ts_model = torch.jit.load(str(model_path), map_location=device)
    ts_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    
    print(f"\nMeasuring TorchScript model latency (CPU, {runs} runs)...")
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            ts_model(dummy_input)
    
    # Measure
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            ts_model(dummy_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'fps': float(1000 / np.mean(latencies))
    }


def measure_latency_onnx(model_path: Path, imgsz: int = 640, warmup: int = 10, runs: int = 100) -> dict:
    """
    Measure inference latency for ONNX model.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("WARNING: onnxruntime not installed. Skipping ONNX latency measurement.")
        return None
    
    # Create ONNX session
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    
    print(f"\nMeasuring ONNX model latency (CPU, {runs} runs)...")
    
    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: dummy_input})
    
    # Measure
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'fps': float(1000 / np.mean(latencies))
    }


def evaluate_model(
    weights: str,
    data: str,
    export_format: str = None,  # None means no export, evaluate original
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = 'cpu',
    conf: float = 0.001,
    iou: float = 0.6,
    verbose: bool = True,
    benchmark_runs: int = 100,
):
    """
    Evaluate model with full metrics. Optionally export to inference format first.
    
    Note: Custom architectures (ODConv, CoordAtt) may not be compatible with 
    TorchScript/ONNX export. Use export_format=None to evaluate the original model.
    """
    weights_path = Path(weights)
    data_path = Path(data) if not Path(data).is_absolute() else Path(data)
    
    # Make data path absolute if needed
    if not data_path.exists():
        # Try relative to repo root
        data_path = REPO_ROOT / data
    
    print(f"\n{'='*60}")
    print("Model Evaluation")
    print(f"{'='*60}")
    print(f"Weights: {weights_path}")
    print(f"Data config: {data_path}")
    print(f"Export format: {export_format if export_format else 'None (original PyTorch)'}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    # Load original model
    print(f"\nLoading model: {weights_path}")
    model = YOLO(str(weights_path))
    
    # Get model size
    model_size_mb = get_model_size(weights_path)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Export model if requested
    eval_model = model
    eval_model_path = weights_path
    exported_size_mb = None
    
    if export_format:
        try:
            output_dir = weights_path.parent
            exported_path = export_model(model, export_format, output_dir)
            exported_size_mb = get_model_size(exported_path)
            print(f"Exported model size: {exported_size_mb:.2f} MB")
            print(f"Size change: {(exported_size_mb - model_size_mb) / model_size_mb * 100:+.1f}%")
            
            # Load exported model for validation
            eval_model = YOLO(str(exported_path))
            eval_model_path = exported_path
        except Exception as e:
            print(f"\n⚠ Export failed: {e}")
            print("Falling back to original model evaluation...")
            eval_model = model
            eval_model_path = weights_path
    
    # Run validation
    print(f"\n{'='*60}")
    print(f"Running Validation on {'Exported' if export_format and exported_size_mb else 'Original'} Model")
    print(f"{'='*60}")
    
    results = eval_model.val(
        data=str(data_path),
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        conf=conf,
        iou=iou,
        verbose=verbose,
        save_json=False,
        plots=True,
    )
    
    # Print results summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Extract metrics
    metrics = results.results_dict
    
    print(f"\n[Overall Metrics]")
    print(f"  mAP@0.5:      {metrics.get('metrics/mAP50(B)', 0) * 100:.2f}%")
    print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0) * 100:.2f}%")
    print(f"  Precision:    {metrics.get('metrics/precision(B)', 0) * 100:.2f}%")
    print(f"  Recall:       {metrics.get('metrics/recall(B)', 0) * 100:.2f}%")
    
    # Per-class results
    if hasattr(results, 'box') and hasattr(results.box, 'ap_class_index'):
        print(f"\n[Per-Class AP@0.5]")
        names = results.names
        ap50 = results.box.ap50
        
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = names.get(cls_idx, f"Class_{cls_idx}")
            print(f"  {cls_name}: {ap50[i] * 100:.2f}%")
    
    # Model size comparison
    print(f"\n[Model Size]")
    print(f"  Original (.pt): {model_size_mb:.2f} MB")
    if exported_size_mb:
        print(f"  Exported ({export_format}): {exported_size_mb:.2f} MB")
    
    # Measure latency on original PyTorch model
    print(f"\n{'='*60}")
    print("LATENCY BENCHMARKING")
    print(f"{'='*60}")
    
    # Measure PyTorch model
    pytorch_latency = measure_latency_pytorch(model, imgsz=imgsz, runs=benchmark_runs)
    
    # Measure exported model if available
    exported_latency = None
    if export_format and exported_size_mb:
        if export_format == 'torchscript':
            exported_latency = measure_latency_torchscript(eval_model_path, imgsz=imgsz, runs=benchmark_runs)
        elif export_format == 'onnx':
            exported_latency = measure_latency_onnx(eval_model_path, imgsz=imgsz, runs=benchmark_runs)
    
    print(f"\n[Latency Results]")
    print(f"  PyTorch Model:")
    print(f"    Mean: {pytorch_latency['mean_ms']:.2f} ± {pytorch_latency['std_ms']:.2f} ms")
    print(f"    FPS:  {pytorch_latency['fps']:.2f}")
    
    if exported_latency:
        print(f"\n  Exported {export_format.upper()}:")
        print(f"    Mean: {exported_latency['mean_ms']:.2f} ± {exported_latency['std_ms']:.2f} ms")
        print(f"    FPS:  {exported_latency['fps']:.2f}")
        
        speedup = pytorch_latency['mean_ms'] / exported_latency['mean_ms']
        print(f"\n  Speedup: {speedup:.2f}x")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"{'='*60}")
    
    return results, eval_model_path


def main():
    parser = argparse.ArgumentParser(description="Export and evaluate YOLO model")
    parser.add_argument(
        "--weights", 
        type=str, 
        required=True,
        help="Path to model weights (.pt file)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML config"
    )
    parser.add_argument(
        "--export",
        type=str,
        default="torchscript",
        choices=["torchscript", "onnx"],
        help="Export format (default: torchscript)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for validation (default: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold (default: 0.001)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS (default: 0.6)"
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of runs for latency benchmark (default: 100)"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        weights=args.weights,
        data=args.data,
        export_format=args.export,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        benchmark_runs=args.benchmark_runs,
    )


if __name__ == "__main__":
    main()
