"""
Detailed model comparison: Size, Speed, and Accuracy
Compares INT8 and FP32 models across multiple dimensions
"""

import sys
from pathlib import Path
import torch
import time
import numpy as np
from contextlib import contextmanager

# Add local ultralytics to path
sys.path.insert(0, str(Path(__file__).parent / 'obc-yolov8' / 'ultralytics10.24'))

from ultralytics import YOLO
from ultralytics.nn.tasks import ensure_module_bookkeeping
from ultralytics.utils import LOGGER
import warnings
warnings.filterwarnings('ignore')


def format_size(size_bytes):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_model_size(model_path):
    """Get model file size and approximate memory footprint"""
    path = Path(model_path)
    if not path.exists():
        return None, None
    
    file_size = path.stat().st_size
    
    # Try to estimate model memory footprint
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model_obj = checkpoint['model']
            if isinstance(model_obj, torch.nn.Module):
                # Count parameters
                param_count = sum(p.numel() for p in model_obj.parameters())
                # Estimate memory (FP32: 4 bytes/param, INT8: 1 byte/param + overhead)
                if hasattr(model_obj, '_packed_params') or any('Quantized' in str(type(m)) for m in model_obj.modules()):
                    # INT8 model
                    memory_estimate = param_count * 1.2  # INT8 + overhead
                else:
                    # FP32 model
                    memory_estimate = param_count * 4
                return file_size, memory_estimate
    except Exception as e:
        LOGGER.debug(f"Could not estimate memory: {e}")
    
    return file_size, None


@contextmanager
def timer():
    """Context manager for timing operations"""
    start = time.time()
    yield lambda: time.time() - start


def benchmark_inference(yolo_model, device, num_runs=50, warmup=10, imgsz=640):
    """Benchmark inference speed using YOLO model's forward pass"""
    yolo_model.model.eval()
    
    # Use YOLO's predict method which handles quantized models properly
    dummy_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(warmup):
        try:
            _ = yolo_model.predict(dummy_image, imgsz=imgsz, device=device, verbose=False)
        except Exception:
            # Fallback to direct model call
            dummy_input = torch.randn(1, 3, imgsz, imgsz)
            if device != 'cpu' and torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                yolo_model.model = yolo_model.model.cuda()
            else:
                dummy_input = dummy_input.cpu()
                yolo_model.model = yolo_model.model.cpu()
            with torch.no_grad():
                _ = yolo_model.model(dummy_input)
    
    # Benchmark
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        try:
            _ = yolo_model.predict(dummy_image, imgsz=imgsz, device=device, verbose=False)
        except Exception:
            # Fallback to direct model call
            dummy_input = torch.randn(1, 3, imgsz, imgsz)
            if device != 'cpu' and torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            else:
                dummy_input = dummy_input.cpu()
            with torch.no_grad():
                _ = yolo_model.model(dummy_input)
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }


def evaluate_model(model_path, data_cfg, device, imgsz=640, batch=16):
    """Evaluate model and return metrics"""
    try:
        # Set backend for INT8 models
        if 'int8' in str(model_path).lower() or 'quantized' in str(model_path).lower():
            if 'fbgemm' in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = 'fbgemm'
        
        model = YOLO(str(model_path))
        
        # Repair bookkeeping if needed
        if hasattr(model.model, 'modules'):
            ensure_module_bookkeeping(model.model, recursive=True)
        
        # Run validation
        results = model.val(
            data=data_cfg,
            imgsz=imgsz,
            batch=batch,
            device=device,
            plots=False,
            save=False,
            verbose=False
        )
        
        if hasattr(results, 'box'):
            return {
                'map50': results.box.map50,
                'map': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0.0,
                'speed': results.speed if hasattr(results, 'speed') else None
            }
    except Exception as e:
        LOGGER.error(f"Evaluation failed: {e}")
        import traceback
        LOGGER.debug(traceback.format_exc())
        return None


def main():
    int8_path = '/home/ubuntu/obc-yolov8/runs/detect/train_qat25/weights/last_int8.pt'
    qat_path = '/home/ubuntu/obc-yolov8/runs/detect/train_qat25/weights/last_qat.pt'
    fp32_path = '/home/ubuntu/obc-yolov8/obc-yolov8/runs/detect/train3/weights/last.pt'
    data_cfg = 'obc-yolov8/ultralytics10.24/ultralytics/cfg/datasets/combined_china_motorbike.yaml'
    imgsz = 640
    batch = 16
    
    print("=" * 100)
    print("DETAILED MODEL COMPARISON: QAT vs INT8 vs FP32")
    print("=" * 100)
    print()
    
    # 1. Model Size Comparison
    print("1. MODEL SIZE COMPARISON")
    print("-" * 100)
    
    int8_file_size, int8_memory = get_model_size(int8_path)
    qat_file_size, qat_memory = get_model_size(qat_path) if Path(qat_path).exists() else (None, None)
    fp32_file_size, fp32_memory = get_model_size(fp32_path)
    
    print(f"QAT Model ({Path(qat_path).name if Path(qat_path).exists() else 'N/A'}):")
    if qat_file_size:
        print(f"  File Size:        {format_size(qat_file_size)}")
        if qat_memory:
            print(f"  Estimated Memory:  {format_size(qat_memory)}")
    else:
        print(f"  File Size:        N/A (file not found)")
    print()
    
    print(f"INT8 Model ({Path(int8_path).name}):")
    print(f"  File Size:        {format_size(int8_file_size) if int8_file_size else 'N/A'}")
    if int8_memory:
        print(f"  Estimated Memory:  {format_size(int8_memory)}")
    print()
    
    print(f"FP32 Model ({Path(fp32_path).name}):")
    print(f"  File Size:        {format_size(fp32_file_size) if fp32_file_size else 'N/A'}")
    if fp32_memory:
        print(f"  Estimated Memory:  {format_size(fp32_memory)}")
    print()
    
    if int8_file_size and fp32_file_size:
        size_reduction = (1 - int8_file_size / fp32_file_size) * 100
        print(f"INT8 vs FP32 Size Reduction: {size_reduction:.2f}% ({format_size(fp32_file_size - int8_file_size)} smaller)")
        if int8_memory and fp32_memory:
            memory_reduction = (1 - int8_memory / fp32_memory) * 100
            print(f"INT8 vs FP32 Memory Reduction: {memory_reduction:.2f}% ({format_size(fp32_memory - int8_memory)} less)")
    if qat_file_size and fp32_file_size:
        qat_size_reduction = (1 - qat_file_size / fp32_file_size) * 100
        print(f"QAT vs FP32 Size Reduction: {qat_size_reduction:.2f}% ({format_size(fp32_file_size - qat_file_size)} smaller)")
    print()
    print()
    
    # 2. Inference Speed Comparison
    print("2. INFERENCE SPEED COMPARISON")
    print("-" * 100)
    
    # Benchmark on CPU (fair comparison for all models)
    print("CPU Inference Speed (50 runs, 10 warmup):")
    print()
    
    # QAT on CPU
    qat_cpu_stats = None
    if Path(qat_path).exists():
        try:
            print("QAT Model (CPU - with FakeQuantize):")
            qat_model = YOLO(qat_path)
            ensure_module_bookkeeping(qat_model.model, recursive=True)
            qat_model.model.eval()
            qat_cpu_stats = benchmark_inference(qat_model, 'cpu', num_runs=50, warmup=10, imgsz=imgsz)
            print(f"  Mean:     {qat_cpu_stats['mean']:.2f} ms")
            print(f"  Median:   {qat_cpu_stats['median']:.2f} ms")
            print(f"  Std Dev:  {qat_cpu_stats['std']:.2f} ms")
            print(f"  Min:      {qat_cpu_stats['min']:.2f} ms")
            print(f"  Max:      {qat_cpu_stats['max']:.2f} ms")
            print(f"  P95:      {qat_cpu_stats['p95']:.2f} ms")
            print(f"  P99:      {qat_cpu_stats['p99']:.2f} ms")
            print(f"  Throughput: {1000/qat_cpu_stats['mean']:.2f} FPS")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            LOGGER.debug(traceback.format_exc())
    
    print()
    
    try:
        # INT8 on CPU
        print("INT8 Model (CPU - fbgemm backend):")
        if 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
        int8_model = YOLO(int8_path)
        ensure_module_bookkeeping(int8_model.model, recursive=True)
        int8_model.model.eval()
        int8_cpu_stats = benchmark_inference(int8_model, 'cpu', num_runs=50, warmup=10, imgsz=imgsz)
        print(f"  Mean:     {int8_cpu_stats['mean']:.2f} ms")
        print(f"  Median:   {int8_cpu_stats['median']:.2f} ms")
        print(f"  Std Dev:  {int8_cpu_stats['std']:.2f} ms")
        print(f"  Min:      {int8_cpu_stats['min']:.2f} ms")
        print(f"  Max:      {int8_cpu_stats['max']:.2f} ms")
        print(f"  P95:      {int8_cpu_stats['p95']:.2f} ms")
        print(f"  P99:      {int8_cpu_stats['p99']:.2f} ms")
        print(f"  Throughput: {1000/int8_cpu_stats['mean']:.2f} FPS")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        LOGGER.debug(traceback.format_exc())
        int8_cpu_stats = None
    
    print()
    
    try:
        # FP32 on CPU
        print("FP32 Model (CPU):")
        fp32_model = YOLO(fp32_path)
        fp32_model.model.eval()
        fp32_cpu_stats = benchmark_inference(fp32_model, 'cpu', num_runs=50, warmup=10, imgsz=imgsz)
        print(f"  Mean:     {fp32_cpu_stats['mean']:.2f} ms")
        print(f"  Median:   {fp32_cpu_stats['median']:.2f} ms")
        print(f"  Std Dev:  {fp32_cpu_stats['std']:.2f} ms")
        print(f"  Min:      {fp32_cpu_stats['min']:.2f} ms")
        print(f"  Max:      {fp32_cpu_stats['max']:.2f} ms")
        print(f"  P95:      {fp32_cpu_stats['p95']:.2f} ms")
        print(f"  P99:      {fp32_cpu_stats['p99']:.2f} ms")
        print(f"  Throughput: {1000/fp32_cpu_stats['mean']:.2f} FPS")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        LOGGER.debug(traceback.format_exc())
        fp32_cpu_stats = None
    
    print()
    
    # CPU Speed Comparison
    if int8_cpu_stats and fp32_cpu_stats:
        speedup = fp32_cpu_stats['mean'] / int8_cpu_stats['mean']
        print(f"CPU Speed Comparison:")
        print(f"  FP32 vs INT8: {speedup:.2f}x (FP32 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'})")
    if qat_cpu_stats and fp32_cpu_stats:
        qat_speedup = fp32_cpu_stats['mean'] / qat_cpu_stats['mean']
        print(f"  FP32 vs QAT: {qat_speedup:.2f}x (FP32 is {qat_speedup:.2f}x {'faster' if qat_speedup > 1 else 'slower'})")
    if qat_cpu_stats and int8_cpu_stats:
        qat_int8_speedup = int8_cpu_stats['mean'] / qat_cpu_stats['mean']
        print(f"  QAT vs INT8: {qat_int8_speedup:.2f}x (QAT is {qat_int8_speedup:.2f}x {'faster' if qat_int8_speedup < 1 else 'slower'})")
    
    print()
    
    # GPU benchmark (FP32 and QAT, INT8 typically runs on CPU)
    if torch.cuda.is_available():
        print("GPU Inference Speed (FP32 and QAT, 50 runs, 10 warmup):")
        print()
        
        # QAT on GPU
        qat_gpu_stats = None
        if Path(qat_path).exists():
            try:
                print("QAT Model (GPU):")
                qat_model = YOLO(qat_path)
                ensure_module_bookkeeping(qat_model.model, recursive=True)
                qat_model.model.eval()
                qat_gpu_stats = benchmark_inference(qat_model, '0', num_runs=50, warmup=10, imgsz=imgsz)
                print(f"  Mean:     {qat_gpu_stats['mean']:.2f} ms")
                print(f"  Median:   {qat_gpu_stats['median']:.2f} ms")
                print(f"  Std Dev:  {qat_gpu_stats['std']:.2f} ms")
                print(f"  Min:      {qat_gpu_stats['min']:.2f} ms")
                print(f"  Max:      {qat_gpu_stats['max']:.2f} ms")
                print(f"  P95:      {qat_gpu_stats['p95']:.2f} ms")
                print(f"  P99:      {qat_gpu_stats['p99']:.2f} ms")
                print(f"  Throughput: {1000/qat_gpu_stats['mean']:.2f} FPS")
                print()
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                LOGGER.debug(traceback.format_exc())
        
        try:
            fp32_model = YOLO(fp32_path)
            fp32_model.model.eval()
            fp32_gpu_stats = benchmark_inference(fp32_model, '0', num_runs=50, warmup=10, imgsz=imgsz)
            print(f"FP32 Model (GPU):")
            print(f"  Mean:     {fp32_gpu_stats['mean']:.2f} ms")
            print(f"  Median:   {fp32_gpu_stats['median']:.2f} ms")
            print(f"  Std Dev:  {fp32_gpu_stats['std']:.2f} ms")
            print(f"  Min:      {fp32_gpu_stats['min']:.2f} ms")
            print(f"  Max:      {fp32_gpu_stats['max']:.2f} ms")
            print(f"  P95:      {fp32_gpu_stats['p95']:.2f} ms")
            print(f"  P99:      {fp32_gpu_stats['p99']:.2f} ms")
            print(f"  Throughput: {1000/fp32_gpu_stats['mean']:.2f} FPS")
            print()
            if int8_cpu_stats:
                gpu_vs_int8 = fp32_gpu_stats['mean'] / int8_cpu_stats['mean']
                print(f"GPU vs INT8-CPU: {gpu_vs_int8:.2f}x (GPU is {gpu_vs_int8:.2f}x {'faster' if gpu_vs_int8 < 1 else 'slower'})")
            if qat_gpu_stats:
                fp32_vs_qat_gpu = fp32_gpu_stats['mean'] / qat_gpu_stats['mean']
                print(f"FP32 vs QAT (GPU): {fp32_vs_qat_gpu:.2f}x (FP32 is {fp32_vs_qat_gpu:.2f}x {'faster' if fp32_vs_qat_gpu < 1 else 'slower'})")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            LOGGER.debug(traceback.format_exc())
    
    print()
    print()
    
    # 3. Accuracy Comparison
    print("3. ACCURACY COMPARISON")
    print("-" * 100)
    
    # Evaluate QAT model
    qat_metrics = None
    if Path(qat_path).exists():
        print("Evaluating QAT model...")
        qat_device = '0' if torch.cuda.is_available() else 'cpu'
        qat_metrics = evaluate_model(qat_path, data_cfg, qat_device, imgsz=imgsz, batch=batch)
    
    print("Evaluating INT8 model...")
    int8_metrics = evaluate_model(int8_path, data_cfg, 'cpu', imgsz=imgsz, batch=batch)
    
    print("Evaluating FP32 model (CPU)...")
    fp32_cpu_metrics = evaluate_model(fp32_path, data_cfg, 'cpu', imgsz=imgsz, batch=batch)
    
    print("Evaluating FP32 model (GPU)...")
    fp32_device = '0' if torch.cuda.is_available() else 'cpu'
    fp32_gpu_metrics = evaluate_model(fp32_path, data_cfg, fp32_device, imgsz=imgsz, batch=batch)
    
    # Use GPU metrics for FP32 if available, otherwise CPU
    fp32_metrics = fp32_gpu_metrics if fp32_gpu_metrics else fp32_cpu_metrics
    
    print()
    
    # Print QAT metrics
    if qat_metrics:
        print("QAT Model Metrics:")
        print(f"  mAP@0.5:      {qat_metrics['map50']:.4f} ({qat_metrics['map50']*100:.2f}%)")
        print(f"  mAP@0.5:0.95: {qat_metrics['map']:.4f} ({qat_metrics['map']*100:.2f}%)")
        print(f"  Precision:    {qat_metrics['precision']:.4f} ({qat_metrics['precision']*100:.2f}%)")
        print(f"  Recall:       {qat_metrics['recall']:.4f} ({qat_metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:     {qat_metrics['f1']:.4f} ({qat_metrics['f1']*100:.2f}%)")
        if qat_metrics['speed']:
            print(f"  Inference:    {qat_metrics['speed'].get('inference', 'N/A'):.2f} ms/img")
        print()
    
    if int8_metrics:
        print("INT8 Model Metrics:")
        print(f"  mAP@0.5:      {int8_metrics['map50']:.4f} ({int8_metrics['map50']*100:.2f}%)")
        print(f"  mAP@0.5:0.95: {int8_metrics['map']:.4f} ({int8_metrics['map']*100:.2f}%)")
        print(f"  Precision:    {int8_metrics['precision']:.4f} ({int8_metrics['precision']*100:.2f}%)")
        print(f"  Recall:       {int8_metrics['recall']:.4f} ({int8_metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:     {int8_metrics['f1']:.4f} ({int8_metrics['f1']*100:.2f}%)")
        if int8_metrics['speed']:
            print(f"  Inference:    {int8_metrics['speed'].get('inference', 'N/A'):.2f} ms/img")
        print()
    
    if fp32_cpu_metrics:
        print("FP32 Model Metrics (CPU):")
        print(f"  mAP@0.5:      {fp32_cpu_metrics['map50']:.4f} ({fp32_cpu_metrics['map50']*100:.2f}%)")
        print(f"  mAP@0.5:0.95: {fp32_cpu_metrics['map']:.4f} ({fp32_cpu_metrics['map']*100:.2f}%)")
        print(f"  Precision:    {fp32_cpu_metrics['precision']:.4f} ({fp32_cpu_metrics['precision']*100:.2f}%)")
        print(f"  Recall:       {fp32_cpu_metrics['recall']:.4f} ({fp32_cpu_metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:     {fp32_cpu_metrics['f1']:.4f} ({fp32_cpu_metrics['f1']*100:.2f}%)")
        if fp32_cpu_metrics['speed']:
            print(f"  Inference:    {fp32_cpu_metrics['speed'].get('inference', 'N/A'):.2f} ms/img")
        print()
    
    if fp32_gpu_metrics:
        print("FP32 Model Metrics (GPU):")
        print(f"  mAP@0.5:      {fp32_gpu_metrics['map50']:.4f} ({fp32_gpu_metrics['map50']*100:.2f}%)")
        print(f"  mAP@0.5:0.95: {fp32_gpu_metrics['map']:.4f} ({fp32_gpu_metrics['map']*100:.2f}%)")
        print(f"  Precision:    {fp32_gpu_metrics['precision']:.4f} ({fp32_gpu_metrics['precision']*100:.2f}%)")
        print(f"  Recall:       {fp32_gpu_metrics['recall']:.4f} ({fp32_gpu_metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:     {fp32_gpu_metrics['f1']:.4f} ({fp32_gpu_metrics['f1']*100:.2f}%)")
        if fp32_gpu_metrics['speed']:
            print(f"  Inference:    {fp32_gpu_metrics['speed'].get('inference', 'N/A'):.2f} ms/img")
        print()
    
    # Accuracy comparisons
    if int8_metrics and fp32_metrics:
        print("Accuracy Differences (INT8 - FP32):")
        map50_diff = int8_metrics['map50'] - fp32_metrics['map50']
        map_diff = int8_metrics['map'] - fp32_metrics['map']
        prec_diff = int8_metrics['precision'] - fp32_metrics['precision']
        recall_diff = int8_metrics['recall'] - fp32_metrics['recall']
        f1_diff = int8_metrics['f1'] - fp32_metrics['f1']
        
        print(f"  mAP@0.5:      {map50_diff:+.4f} ({map50_diff*100:+.2f}%)")
        print(f"  mAP@0.5:0.95: {map_diff:+.4f} ({map_diff*100:+.2f}%)")
        print(f"  Precision:    {prec_diff:+.4f} ({prec_diff*100:+.2f}%)")
        print(f"  Recall:       {recall_diff:+.4f} ({recall_diff*100:+.2f}%)")
        print(f"  F1-Score:     {f1_diff:+.4f} ({f1_diff*100:+.2f}%)")
        print()
        
        if fp32_metrics['map50'] > 0:
            retention = (int8_metrics['map50'] / fp32_metrics['map50']) * 100
            print(f"INT8 Accuracy Retention: {retention:.2f}% (mAP@0.5)")
        print()
    
    if qat_metrics and fp32_metrics:
        print("Accuracy Differences (QAT - FP32):")
        qat_map50_diff = qat_metrics['map50'] - fp32_metrics['map50']
        qat_map_diff = qat_metrics['map'] - fp32_metrics['map']
        qat_prec_diff = qat_metrics['precision'] - fp32_metrics['precision']
        qat_recall_diff = qat_metrics['recall'] - fp32_metrics['recall']
        qat_f1_diff = qat_metrics['f1'] - fp32_metrics['f1']
        
        print(f"  mAP@0.5:      {qat_map50_diff:+.4f} ({qat_map50_diff*100:+.2f}%)")
        print(f"  mAP@0.5:0.95: {qat_map_diff:+.4f} ({qat_map_diff*100:+.2f}%)")
        print(f"  Precision:    {qat_prec_diff:+.4f} ({qat_prec_diff*100:+.2f}%)")
        print(f"  Recall:       {qat_recall_diff:+.4f} ({qat_recall_diff*100:+.2f}%)")
        print(f"  F1-Score:     {qat_f1_diff:+.4f} ({qat_f1_diff*100:+.2f}%)")
        print()
        
        if fp32_metrics['map50'] > 0:
            qat_retention = (qat_metrics['map50'] / fp32_metrics['map50']) * 100
            print(f"QAT Accuracy Retention: {qat_retention:.2f}% (mAP@0.5)")
        print()
    
    if qat_metrics and int8_metrics:
        print("Accuracy Differences (INT8 - QAT):")
        int8_qat_map50_diff = int8_metrics['map50'] - qat_metrics['map50']
        int8_qat_map_diff = int8_metrics['map'] - qat_metrics['map']
        int8_qat_prec_diff = int8_metrics['precision'] - qat_metrics['precision']
        int8_qat_recall_diff = int8_metrics['recall'] - qat_metrics['recall']
        int8_qat_f1_diff = int8_metrics['f1'] - qat_metrics['f1']
        
        print(f"  mAP@0.5:      {int8_qat_map50_diff:+.4f} ({int8_qat_map50_diff*100:+.2f}%)")
        print(f"  mAP@0.5:0.95: {int8_qat_map_diff:+.4f} ({int8_qat_map_diff*100:+.2f}%)")
        print(f"  Precision:    {int8_qat_prec_diff:+.4f} ({int8_qat_prec_diff*100:+.2f}%)")
        print(f"  Recall:       {int8_qat_recall_diff:+.4f} ({int8_qat_recall_diff*100:+.2f}%)")
        print(f"  F1-Score:     {int8_qat_f1_diff:+.4f} ({int8_qat_f1_diff*100:+.2f}%)")
        print()
    
    print()
    print()
    
    # 4. Summary
    print("4. SUMMARY")
    print("-" * 100)
    
    if int8_file_size and fp32_file_size:
        size_reduction = (1 - int8_file_size / fp32_file_size) * 100
        print(f"✓ Model Size: INT8 is {size_reduction:.1f}% smaller than FP32")
    if qat_file_size and fp32_file_size:
        qat_size_reduction = (1 - qat_file_size / fp32_file_size) * 100
        print(f"✓ Model Size: QAT is {qat_size_reduction:.1f}% smaller than FP32")
    if int8_cpu_stats and fp32_cpu_stats:
        speedup = fp32_cpu_stats['mean'] / int8_cpu_stats['mean']
        print(f"✓ CPU Speed: FP32 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than INT8 on CPU")
    if qat_cpu_stats and fp32_cpu_stats:
        qat_speedup = fp32_cpu_stats['mean'] / qat_cpu_stats['mean']
        print(f"✓ CPU Speed: FP32 is {qat_speedup:.2f}x {'faster' if qat_speedup > 1 else 'slower'} than QAT on CPU")
    if int8_metrics and fp32_metrics:
        if fp32_metrics['map50'] > 0:
            retention = (int8_metrics['map50'] / fp32_metrics['map50']) * 100
            print(f"✓ Accuracy: INT8 retains {retention:.1f}% of FP32 accuracy (mAP@0.5)")
    if qat_metrics and fp32_metrics:
        if fp32_metrics['map50'] > 0:
            qat_retention = (qat_metrics['map50'] / fp32_metrics['map50']) * 100
            print(f"✓ Accuracy: QAT retains {qat_retention:.1f}% of FP32 accuracy (mAP@0.5)")
    
    print()
    print("=" * 100)


if __name__ == '__main__':
    main()

