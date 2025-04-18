from ultralytics import YOLO

# Path to your pre-trained model (you can change this to other YOLOv8 models like yolov8s.pt or yolov8m.pt)
model_path = 'yolov8n.pt'

# Path to your custom dataset YAML file
data_yaml = 'D:/ML/Automated Face Mask Detection/face-mask-dataset/data.yaml'


# Define training parameters
epochs = 50  # Number of epochs to train
img_size = 450 # Image size for training

# Create and train the model
model = YOLO(model_path)

# Start training the model
model.train(data=data_yaml, epochs=epochs, imgsz=img_size)

# After training, the best model weights will be saved in the 'runs/detect/train' directory
