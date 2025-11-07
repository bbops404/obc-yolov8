
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def main():
    model = YOLO("ultralytics/cfg/models/v8/yolov8-CA.yaml")
    results = model.train(data="ultralytics/cfg/datasets/combined_china_motorbike.yaml",
                          epochs=300, imgsz=640, device=0)

if __name__ == '__main__':
    main()
