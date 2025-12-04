import os
import shutil
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# === PATHS ===
BASE_DIR = "/Users/user/Documents/obc-yolov8/ultralytics10.24/dataset_root/combined_all"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
OUT_DIR = os.path.join(BASE_DIR, "folds")

os.makedirs(OUT_DIR, exist_ok=True)

# === Load dataset ===
images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))])

labels = []
valid_images = []

for img in images:
    label_path = os.path.join(LBL_DIR, img.replace(".jpg", ".txt").replace(".png", ".txt"))
    if os.path.exists(label_path):
        with open(label_path) as f:
            cls = f.readline().split()[0]  # first class in file
            labels.append(int(cls))
            valid_images.append(img)

images = valid_images

print(f"Total usable images: {len(images)}")

# === Stratified 5-Fold Split ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels), 1):
    print(f"\nCreating Fold {fold}")

    fold_dir = os.path.join(OUT_DIR, f"fold{fold}")
    train_img_dir = os.path.join(fold_dir, "train/images")
    train_lbl_dir = os.path.join(fold_dir, "train/labels")
    val_img_dir = os.path.join(fold_dir, "val/images")
    val_lbl_dir = os.path.join(fold_dir, "val/labels")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # === TRAIN COPY ===
    for idx in tqdm(train_idx, desc=f"Fold {fold} Train"):
        img = images[idx]
        lbl = img.replace(".jpg", ".txt").replace(".png", ".txt")

        shutil.copy(os.path.join(IMG_DIR, img), os.path.join(train_img_dir, img))
        shutil.copy(os.path.join(LBL_DIR, lbl), os.path.join(train_lbl_dir, lbl))

    # === VAL COPY ===
    for idx in tqdm(val_idx, desc=f"Fold {fold} Val"):
        img = images[idx]
        lbl = img.replace(".jpg", ".txt").replace(".png", ".txt")

        shutil.copy(os.path.join(IMG_DIR, img), os.path.join(val_img_dir, img))
        shutil.copy(os.path.join(LBL_DIR, lbl), os.path.join(val_lbl_dir, lbl))

print("\n✅ Stratified 5-Fold splitting completed successfully.")
