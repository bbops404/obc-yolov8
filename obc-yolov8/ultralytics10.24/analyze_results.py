import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def analyze_training_results():
    """Analyze the training results and generate comprehensive reports"""
    
    # Try to find the most recent training directory
    base_dir = "/home/ubuntu/obc-yolov8/OBC-YOLOv8/runs/detect"
    results_dir = None
    
    if os.path.exists(base_dir):
        # Find all training directories
        train_dirs = [d for d in os.listdir(base_dir) if d.startswith('train') and os.path.isdir(os.path.join(base_dir, d))]
        if train_dirs:
            # Sort by directory name to get the most recent
            train_dirs.sort()
            results_dir = os.path.join(base_dir, train_dirs[-1])
        else:
            results_dir = "/home/ubuntu/obc-yolov8/OBC-YOLOv8/runs/detect/train4"
    else:
        results_dir = "/home/ubuntu/obc-yolov8/OBC-YOLOv8/runs/detect/train4"
    
    print("=== Training Results Analysis ===")
    print(f"Results directory: {results_dir}")
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print("❌ Results directory not found!")
        print(f"Available directories in {base_dir}:")
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                print(f"  - {item}")
        return
    
    # List all files in results directory
    print("\n📁 Files in results directory:")
    for file in os.listdir(results_dir):
        file_path = os.path.join(results_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # Size in MB
            print(f"  - {file} ({size:.1f} MB)")
        else:
            print(f"  - {file}/ (directory)")
    
    # Check for model weights
    weights_dir = os.path.join(results_dir, "weights")
    if os.path.exists(weights_dir):
        print(f"\n🎯 Model weights found:")
        for weight_file in os.listdir(weights_dir):
            weight_path = os.path.join(weights_dir, weight_file)
            size = os.path.getsize(weight_path) / (1024*1024)
            print(f"  - {weight_file} ({size:.1f} MB)")
    
    # Check for training arguments
    args_file = os.path.join(results_dir, "args.yaml")
    if os.path.exists(args_file):
        print(f"\n⚙️ Training configuration:")
        with open(args_file, 'r') as f:
            args = yaml.safe_load(f)
            for key, value in args.items():
                print(f"  - {key}: {value}")
    
    # Check for results CSV
    results_csv = os.path.join(results_dir, "results.csv")
    if os.path.exists(results_csv):
        print(f"\n📊 Training metrics:")
        df = pd.read_csv(results_csv)
        print(f"  - Total epochs: {len(df)}")
        
        # Debug: Print column names to see what's available
        print(f"  - Available columns: {list(df.columns)}")
        
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        
        # Try to access metrics with error handling
        try:
            if 'metrics/mAP50(B)' in df.columns:
                print(f"  - Final mAP50: {df['metrics/mAP50(B)'].iloc[-1]:.3f}")
            else:
                print("  - mAP50 column not found")
                
            if 'metrics/mAP50-95(B)' in df.columns:
                print(f"  - Final mAP50-95: {df['metrics/mAP50-95(B)'].iloc[-1]:.3f}")
            else:
                print("  - mAP50-95 column not found")
                
            if 'metrics/precision(B)' in df.columns:
                print(f"  - Final precision: {df['metrics/precision(B)'].iloc[-1]:.3f}")
            else:
                print("  - Precision column not found")
                
            if 'metrics/recall(B)' in df.columns:
                print(f"  - Final recall: {df['metrics/recall(B)'].iloc[-1]:.3f}")
            else:
                print("  - Recall column not found")
                
        except Exception as e:
            print(f"  - Error accessing metrics: {e}")
        
        # Plot training curves
        plot_training_curves(df, results_dir)
    
    print(f"\n✅ Analysis complete! Check {results_dir} for detailed results.")

def plot_training_curves(df, results_dir):
    """Plot training curves"""
    try:
        plt.figure(figsize=(15, 10))
        
        # mAP curves
        plt.subplot(2, 3, 1)
        if 'metrics/mAP50(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='blue')
        if 'metrics/mAP50-95(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('mAP Curves')
        plt.legend()
        plt.grid(True)
        
        # Precision and Recall
        plt.subplot(2, 3, 2)
        if 'metrics/precision(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='green')
        if 'metrics/recall(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision & Recall')
        plt.legend()
        plt.grid(True)
        
        # Loss curves
        plt.subplot(2, 3, 3)
        if 'train/box_loss' in df.columns:
            plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='purple')
        if 'train/cls_loss' in df.columns:
            plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='brown')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(2, 3, 4)
        if 'lr/pg0' in df.columns:
            plt.plot(df['epoch'], df['lr/pg0'], label='LR', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        # Validation losses
        plt.subplot(2, 3, 5)
        if 'val/box_loss' in df.columns:
            plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='purple')
        if 'val/cls_loss' in df.columns:
            plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', color='brown')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Losses')
        plt.legend()
        plt.grid(True)
        
        # DFL loss
        plt.subplot(2, 3, 6)
        if 'train/dfl_loss' in df.columns:
            plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', color='pink')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DFL Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved to {results_dir}/training_curves.png")
        
    except Exception as e:
        print(f"⚠️ Could not generate training curves: {e}")

if __name__ == '__main__':
    analyze_training_results()
