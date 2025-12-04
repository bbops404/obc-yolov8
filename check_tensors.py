import torch
from ultralytics import YOLO

# Load YOLOv8-CA model with pretrained FP32 weights
weights_path = '/Users/user/Documents/obc-yolov8/obc-yolov8/runs/detect/train3/weights/last.pt'
model = YOLO('/Users/user/Documents/obc-yolov8/ultralytics10.24/ultralytics/cfg/models/v8/yolov8-CA.yaml')
model.load(weights_path)
model.to('cpu')

print("===== FP32 Weights of Selected Layers =====\n")

# Selected layers
selected_layer_names = ['model.10', 'model.21', 'model.23', 'model.24']

def print_tensor_clean(name, tensor):
    # Flatten to 2D if needed (out_channels, -1)
    t_flat = tensor.view(tensor.size(0), -1)
    print(f"{name} - Parameter containing:")
    print(t_flat)
    print(tensor.dtype)
    print()

# Print selected layers
for name, layer in model.model.named_modules():
    if name in selected_layer_names and hasattr(layer, 'weight') and layer.weight is not None:
        print_tensor_clean(name, layer.weight.data)

# Detect head: first few convs
detect_layer = [l for l in model.model.modules() if l.__class__.__name__ == 'Detect'][0]
for idx, conv_seq in enumerate(detect_layer.cv2[:2]):  # first two conv sequences
    for sub_idx, sublayer in enumerate(conv_seq):
        if hasattr(sublayer, 'weight') and sublayer.weight is not None:
            print_tensor_clean(f"", sublayer.weight.data)
