import warnings
import sys
from pathlib import Path

# Ensure we use the local Ultralytics repo (with CA_Attention, ODConv, etc.)
LOCAL_ULTRA_ROOT = Path("/home/ubuntu/obc-yolov8/obc-yolov8/ultralytics10.24")
sys.path.insert(0, str(LOCAL_ULTRA_ROOT))

from ultralytics import YOLO  # now resolves to local ultralytics10.24

warnings.filterwarnings('ignore')

# run command : python train_fp32_fold.py 1  # for fold 1

fold = int(sys.argv[1])  # Pass fold number as argument


def main():
    # Use model config from the local ultralytics10.24 repo
    model_cfg = LOCAL_ULTRA_ROOT / "ultralytics/cfg/models/v8/yolov8-CA.yaml"
    model = YOLO(str(model_cfg))
    # Use local Linux path and per-fold YAML created earlier
    yaml_path = f"/home/ubuntu/obc-yolov8/obc-yolov8/ultralytics10.24/dataset_root/combined_all/fold{fold}.yaml"
    results = model.train(
        data=yaml_path,
        epochs=1,  # quick test run for saving
        imgsz=640,
        device=0
    )


if __name__ == '__main__':
    main()
