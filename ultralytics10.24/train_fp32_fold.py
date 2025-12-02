import warnings
import sys

# Add path to your custom modules so YOLO can see them
sys.path.append('/home/ubuntu/obc-yolov8/obc-yolov8/ultralytics10.24/ultralytics/nn')

# Import custom modules used in your YAML
from ultralytics.nn.ODConv import ODConv
from ultralytics.nn.BoTNet import BoTNet
from ultralytics.nn.CA_Attention import CoordAtt

# Import YOLO after your custom modules
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# Get fold number from command-line argument
if len(sys.argv) < 2:
    raise ValueError("Please provide the fold number as a command-line argument, e.g., python train_fp32_fold.py 1")
fold = int(sys.argv[1])

def main():
    # Use fold-specific dataset YAML
    data_yaml = f"/home/ubuntu/obc-yolov8/obc-yolov8/ultralytics10.24/dataset_root/combined_all/fold{fold}.yaml"
    
    # Load your custom YOLO model
    model = YOLO("/home/ubuntu/obc-yolov8/obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml")
    
    # Train
    results = model.train(
        
        data=data_yaml,
        epochs=300,
        imgsz=640,
        device=0
    )

if __name__ == '__main__':
    main()
