from ultralytics import YOLO
import warnings
import sys

warnings.filterwarnings('ignore')

# run command : python train_fp32_fold.py 1 # for fold 1 


fold = int(sys.argv[1])  # Pass fold number as argument

def main():
    model = YOLO("ultralytics/cfg/models/v8/yolov8-CA.yaml")
    yaml_path = f"/Users/user/Documents/obc-yolov8/ultralytics10.24/dataset_root/combined_all/folds/fold{fold}/fold{fold}.yaml"
    results = model.train(
        data=yaml_path,
        epochs=300,
        imgsz=640,
        device=0
    )

if __name__ == '__main__':
    main()
