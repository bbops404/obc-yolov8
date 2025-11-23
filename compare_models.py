#!/usr/bin/env python3
"""Compare multiple INT8 models by evaluating them."""

import sys
from pathlib import Path

# Add path for evaluate_ptq
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from evaluate_ptq import evaluate_int8_model, DEFAULT_DATA_CFG
import re

# Models to compare
MODELS = {
    "Hybrid Try2": "/home/ubuntu/obc-yolov8/obc-yolov8/runs/detect/hybrid_try2/weights/last_int8.pt",
    "Hybrid QAT CoordAtt": "/home/ubuntu/obc-yolov8/obc-yolov8/runs/detect/hybrid_qat_coordatt/weights/last_int8.pt",
    "Hybrid CoordAtt BotNet": "/home/ubuntu/obc-yolov8/obc-yolov8/runs/detect/train_hybrid_coordatt_botnet/weights/last_int8.pt",
}

def extract_map(eval_results):
    """Extract mAP from evaluation results."""
    if eval_results is None:
        return None, None, None, None
    
    map50 = None
    map = None
    precision = None
    recall = None
    
    # Method 1: box attribute
    if hasattr(eval_results, 'box'):
        map = getattr(eval_results.box, 'map', None)
        map50 = getattr(eval_results.box, 'map50', None)
        precision = getattr(eval_results.box, 'precision', None)
        recall = getattr(eval_results.box, 'recall', None)
    
    # Method 2: Direct attributes
    if map is None and hasattr(eval_results, 'map'):
        map = eval_results.map
    if map50 is None and hasattr(eval_results, 'map50'):
        map50 = eval_results.map50
    if precision is None and hasattr(eval_results, 'precision'):
        precision = eval_results.precision
    if recall is None and hasattr(eval_results, 'recall'):
        recall = eval_results.recall
    
    # Method 3: Metrics dict
    if map is None and hasattr(eval_results, 'metrics'):
        metrics = eval_results.metrics
        if isinstance(metrics, dict):
            map = metrics.get('map', None) or metrics.get('mAP50-95(B)', None)
            map50 = metrics.get('map50', None) or metrics.get('mAP50(B)', None)
            precision = metrics.get('precision', None) or metrics.get('precision(B)', None)
            recall = metrics.get('recall', None) or metrics.get('recall(B)', None)
    
    # Method 4: Dict access
    if map is None and isinstance(eval_results, dict):
        map = eval_results.get('map', None) or eval_results.get('mAP50-95(B)', None)
        map50 = eval_results.get('map50', None) or eval_results.get('mAP50(B)', None)
        precision = eval_results.get('precision', None) or eval_results.get('precision(B)', None)
        recall = eval_results.get('recall', None) or eval_results.get('recall(B)', None)
    
    return map, map50, precision, recall

def extract_speed_from_output(output_text):
    """Extract speed information from YOLO output text."""
    # Look for "Speed: X.Xms preprocess, Y.Yms inference, ..."
    speed_pattern = r'Speed:\s*([\d.]+)\s*ms\s*preprocess,\s*([\d.]+)\s*ms\s*inference'
    match = re.search(speed_pattern, output_text)
    if match:
        preprocess_ms = float(match.group(1))
        inference_ms = float(match.group(2))
        fps = 1000.0 / inference_ms if inference_ms > 0 else 0
        return {
            'preprocess_ms': preprocess_ms,
            'inference_ms': inference_ms,
            'fps': fps
        }
    return None

def main():
    import io
    import contextlib
    
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print()
    
    results = {}
    
    for model_name, model_path in MODELS.items():
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            print(f"⚠️  {model_name}: Model not found at {model_path}")
            results[model_name] = None
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'='*80}\n")
        
        try:
            # Capture stdout to extract speed info
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                eval_data = evaluate_int8_model(
                    int8_weights=model_path,
                    data_cfg=DEFAULT_DATA_CFG,
                    imgsz=640,
                    batch=16,
                    device="cpu",
                    backend="qnnpack"
                )
            
            # Get captured output
            captured_output = output_buffer.getvalue()
            
            # Extract metrics
            eval_results = eval_data.get('eval_results') if isinstance(eval_data, dict) else eval_data
            map, map50, precision, recall = extract_map(eval_results)
            
            # Extract speed info
            speed_info = None
            if isinstance(eval_data, dict):
                speed_info = eval_data.get('speed')
            
            # Get quantization stats
            quant_stats = eval_data.get('quant_stats') if isinstance(eval_data, dict) else None
            model_size_mb = eval_data.get('model_size_mb') if isinstance(eval_data, dict) else None
            quantized_pct = eval_data.get('quantized_pct') if isinstance(eval_data, dict) else None
            
            # Calculate FPS from inference time
            fps = None
            inference_ms = None
            if speed_info:
                if isinstance(speed_info, dict):
                    # YOLO speed dict has 'inference' key with time in ms
                    inference_ms = speed_info.get('inference')
                    if inference_ms is not None and inference_ms > 0:
                        fps = 1000.0 / inference_ms
            
            # If not found in speed_info, try to extract from output text
            if inference_ms is None:
                speed_from_output = extract_speed_from_output(captured_output)
                if speed_from_output:
                    inference_ms = speed_from_output.get('inference_ms')
                    fps = speed_from_output.get('fps')
            
            results[model_name] = {
                'map': map,
                'map50': map50,
                'precision': precision,
                'recall': recall,
                'fps': fps,
                'inference_ms': inference_ms,
                'model_size_mb': model_size_mb,
                'quantized_pct': quantized_pct,
                'quant_stats': quant_stats,
                'path': model_path
            }
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = None
    
    # Print comprehensive comparison table
    print("\n" + "=" * 120)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 120)
    print()
    
    # Table header
    print(f"{'Model':<35} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'Precision':<10} {'Recall':<10} {'FPS':<8} {'Size(MB)':<10} {'Quant%':<8}")
    print("-" * 120)
    
    for model_name, result in results.items():
        if result is None:
            print(f"{model_name:<35} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<10} {'N/A':<8}")
        else:
            map50_str = f"{result['map50']:.4f}" if result['map50'] is not None else "N/A"
            map_str = f"{result['map']:.4f}" if result['map'] is not None else "N/A"
            precision_str = f"{result['precision']:.4f}" if result['precision'] is not None else "N/A"
            recall_str = f"{result['recall']:.4f}" if result['recall'] is not None else "N/A"
            fps_str = f"{result['fps']:.2f}" if result['fps'] is not None else "N/A"
            size_str = f"{result['model_size_mb']:.2f}" if result['model_size_mb'] is not None else "N/A"
            quant_str = f"{result['quantized_pct']:.1f}%" if result['quantized_pct'] is not None else "N/A"
            
            print(f"{model_name:<35} {map50_str:<10} {map_str:<12} {precision_str:<10} {recall_str:<10} {fps_str:<8} {size_str:<10} {quant_str:<8}")
    
    print("-" * 120)
    
    # Detailed analysis
    valid_results = {k: v for k, v in results.items() if v is not None and v['map'] is not None}
    if valid_results:
        print("\n" + "=" * 120)
        print("DETAILED ANALYSIS")
        print("=" * 120)
        
        # Print detailed info for each model
        for model_name, result in valid_results.items():
            print(f"\n{model_name}:")
            print(f"  mAP@0.5:      {result['map50']:.4f}" if result['map50'] is not None else "  mAP@0.5:      N/A")
            print(f"  mAP@0.5:0.95: {result['map']:.4f}" if result['map'] is not None else "  mAP@0.5:0.95: N/A")
            print(f"  Precision:    {result['precision']:.4f}" if result['precision'] is not None else "  Precision:    N/A")
            print(f"  Recall:       {result['recall']:.4f}" if result['recall'] is not None else "  Recall:       N/A")
            print(f"  FPS:          {result['fps']:.2f}" if result['fps'] is not None else "  FPS:          N/A")
            print(f"  Size:         {result['model_size_mb']:.2f} MB" if result['model_size_mb'] is not None else "  Size:         N/A")
            print(f"  Quantized:    {result['quantized_pct']:.1f}%" if result['quantized_pct'] is not None else "  Quantized:    N/A")
        
        # Find best models by different metrics
        print(f"\nBest Models:")
        if valid_results:
            best_map = max(valid_results.items(), key=lambda x: x[1]['map'] if x[1]['map'] else 0)
            print(f"  Best mAP@0.5:0.95: {best_map[0]} ({best_map[1]['map']:.4f})")
            
            best_fps = max([(k, v) for k, v in valid_results.items() if v['fps'] is not None], 
                          key=lambda x: x[1]['fps'], default=None)
            if best_fps:
                print(f"  Best FPS:         {best_fps[0]} ({best_fps[1]['fps']:.2f} FPS)")
            
            smallest_size = min([(k, v) for k, v in valid_results.items() if v['model_size_mb'] is not None], 
                               key=lambda x: x[1]['model_size_mb'], default=None)
            if smallest_size:
                print(f"  Smallest Size:    {smallest_size[0]} ({smallest_size[1]['model_size_mb']:.2f} MB)")
        
        # Pairwise comparisons
        if len(valid_results) > 1:
            print(f"\nPairwise Comparisons:")
            model_list = list(valid_results.items())
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    name1, result1 = model_list[i]
                    name2, result2 = model_list[j]
                    
                    if result1['map'] and result2['map']:
                        map_diff = (result2['map'] - result1['map']) * 100
                        print(f"\n  {name2} vs {name1}:")
                        print(f"    mAP@0.5:0.95: {map_diff:+.2f}%")
                        
                        if result1['fps'] and result2['fps']:
                            fps_change = ((result2['fps'] - result1['fps']) / result1['fps']) * 100
                            print(f"    FPS:         {fps_change:+.2f}%")
                        
                        if result1['model_size_mb'] and result2['model_size_mb']:
                            size_change = ((result2['model_size_mb'] - result1['model_size_mb']) / result1['model_size_mb']) * 100
                            print(f"    Size:        {size_change:+.2f}%")
    
    print("=" * 120)

if __name__ == "__main__":
    main()

