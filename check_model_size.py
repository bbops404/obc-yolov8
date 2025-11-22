import torch
import sys
import torch.nn as nn
from pathlib import Path
import logging

# We will use standard print() for the summary instead of the logger
# LOGGER = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# Set the quantization backend for macOS compatibility
torch.backends.quantized.engine = 'qnnpack' 

# --- PATH LOGIC ---
REPO_ROOT = Path(__file__).parent 
ULTRALYTICS_PARENT_PATH = REPO_ROOT / "ultralytics10.24"

if str(ULTRALYTICS_PARENT_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PARENT_PATH))
# --- END PATH LOGIC ---

try:
    from ultralytics.nn.ODConv import ODConv 
    # Also import the base YOLO class we need for loading
    from ultralytics import YOLO 
except ImportError as e:
    print(f"ERROR: Failed to import required modules (ODConv, YOLO). Error: {e}")
    sys.exit(1)


def inspect_quantized_model(model):
    """
    Safely inspects a potentially packed quantized model and lists quantized layers.
    Returns a dictionary of found quantized modules.
    """
    quantized_modules = {}

    def _recursive_inspect(module, name_prefix=''):
        # Iterate safely over children
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            module_type = type(child).__name__
            module_path = type(child).__module__

            is_quantized = 'quantized' in module_path.lower() or 'Quantized' in module_type

            if is_quantized and 'activation' not in module_type.lower():
                quantized_modules[full_name] = module_type
            
            # Recursively call this function if the child has children (safely)
            if hasattr(child, '_modules') and len(child._modules) > 0:
                _recursive_inspect(child, full_name)

    _recursive_inspect(model)
    return quantized_modules


def print_quantized_layers_summary(quantized_modules):
    # Use standard print statements to force output
    print("=" * 80)
    print("QUANTIZED LAYERS SUMMARY")
    print("=" * 80)
    print(f"Total identified quantized layers: {len(quantized_modules)}")
    
    if quantized_modules:
        print("\nQuantized layers found:")
        print("-" * 80)
        # Sort keys for consistent output
        for i, name in enumerate(sorted(quantized_modules.keys()), 1):
            layer_type = quantized_modules[name]
            print(f"  {i:3d}. {name:60s}  [{layer_type}]")
    print("=" * 80)


# --- Main execution logic ---

model_path = "/Users/user/Documents/obc-yolov8/runs/detect/Full_PTQ/weights/int8.pt" 

try:
    model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    print(f"\nKeys found in '{model_path}':")
    print(model_data.keys())
    print("-" * 40)

    if 'model' in model_data:
        int8_model = model_data['model']
        print(f"Successfully loaded full model object (Backend: {model_data.get('backend', 'N/A')}).")
        
        # Run the safe inspection function with standard print
        found_layers = inspect_quantized_model(int8_model)
        print_quantized_layers_summary(found_layers)

    else:
        print("Model file only contains state_dict. Cannot inspect layers without rebuilding the architecture.")
        

except FileNotFoundError as e:
    print(f"Error: Model file not found. Check paths.\nDetails: {e}")
except Exception as e:
    print(f"AN UNEXPECTED ERROR OCCURRED: {e}")
    sys.exit(1)

