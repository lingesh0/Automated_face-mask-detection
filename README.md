🧠 Automated Face Mask Detection using YOLOv5
This project detects whether people are wearing a face mask, not wearing a mask, or wearing it incorrectly using a custom-trained YOLOv5 model. It includes automatic labeling, training, and real-time detection.

📁 Directory Structure

Automated Face Mask Detection/
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   ├── val/
│   │   ├── images/
│
├── data.yaml
├── train.py
├── detect.py
├── requirements.txt
├── runs/
├── yolov5s.pt
└── README.md
⚙️ 1. Setup Instructions
✅ Clone YOLOv5

git clone https://github.com/ultralytics/yolov5
cd yolov5
✅ Create a Virtual Environment (Optional but Recommended)

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
✅ Install Dependencies

pip install -r requirements.txt
pip install ultralytics
📦 2. Dataset Setup
Place raw unlabeled images in:

dataset/train/images/ → for mask images

dataset/val/images/ → for no_mask images

✨ 3. Auto-Label Images
Use a pretrained YOLOv5 model to automatically generate labels.


python train.py
The script:

Loads yolov5s or yolov5su model

Auto-labels images using model.predict()

Saves labels in dataset/train/labels/ and dataset/val/labels/

📄 4. data.yaml Configuration

path: dataset
train: train/images
val: val/images

names:
  0: mask
  1: no_mask
🏋️ 5. Train YOLOv5 Model

from ultralytics import YOLO

model = YOLO('yolov5su.pt')
model.train(data='dataset/data.yaml', epochs=20, imgsz=416)
📷 6. Run Inference

model = YOLO("runs/detect/train/weights/best.pt")
model.predict("path/to/test/image.jpg", save=True, conf=0.5)
🖥️ 7. Run Real-Time Webcam Detection

model.predict(source=0, show=True)
🚨 Issues & Fixes
No Labels Warning: Ensure labels are saved in dataset/train/labels/ and dataset/val/labels/.

RuntimeError in validation: Happens if labels are missing or corrupted.

Improve predictions: Manually review and fix bad auto-labels.

📊 Output
Training logs & metrics: runs/detect/train/

Predicted images: runs/detect/predict/

🧪 Tested On
Windows 10

Python 3.10

YOLOv5 7.0+

PyTorch 2.0+

📬 Contact
For queries, feel free to reach out!
