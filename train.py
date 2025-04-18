import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# Paths
base_path = r"D:\ML\Automated Face Mask Detection\dataset"
train_images_path = os.path.join(base_path, "train", "images")
val_images_path = os.path.join(base_path, "val", "images")
train_labels_path = os.path.join(base_path, "train", "labels")
val_labels_path = os.path.join(base_path, "val", "labels")

# Create label folders
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# Load pre-trained model
model = YOLO("yolov5s.pt")

# Function to auto-label
def auto_label_images(image_dir, label_dir):
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        results = model(img_path)  # Predict

        label_file = os.path.join(label_dir, Path(img_name).stem + ".txt")
        with open(label_file, 'w') as f:
            for result in results:
                for box in result.boxes:
                    cls = 0 if "train" in image_dir else 1  # 0: mask, 1: no_mask
                    x_center, y_center, width, height = box.xywh[0].tolist()
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]
                    # Normalize
                    x_center /= w
                    y_center /= h
                    width /= w
                    height /= h
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Labeled {len(images)} images in {image_dir}")

# Label train and val sets
auto_label_images(train_images_path, train_labels_path)
auto_label_images(val_images_path, val_labels_path)

# YAML config
import yaml
data_yaml = {
    'train': os.path.join(base_path, 'train'),
    'val': os.path.join(base_path, 'val'),
    'nc': 2,
    'names': ['mask', 'no_mask']
}
with open(os.path.join(base_path, 'data.yaml'), 'w') as f:
    yaml.dump(data_yaml, f)

# Train your custom model
custom_model = YOLO("yolov5s.pt")
custom_model.train(data=os.path.join(base_path, "data.yaml"), epochs=20, imgsz=400)
