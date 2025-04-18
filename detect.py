import cv2
from ultralytics import YOLO
import os


model_path = 'D:\\ML\\Automated Face Mask Detection\\runs\\detect\\train\\weights\\best.pt'
model = YOLO(model_path)  

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a directory to save captured images if it doesn't exist
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')

# Define a function to save images with no mask
def save_image(frame, index):
    filename = f'captured_images/no_mask_{index}.jpg'
    cv2.imwrite(filename, frame)

index = 0  # Counter to save unique images

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Perform prediction on the current frame using the YOLO model
    results = model(frame)  # Directly pass the frame to the YOLO model

    # Process the predictions (bounding boxes, labels, and scores)
    for result in results:
        boxes = result.boxes.xyxy  # (x1, y1, x2, y2) for bounding boxes
        confidences = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class indices (e.g., mask or no mask)
        labels = result.names  # Class names (e.g., mask, no mask)

        # Iterate through the boxes and draw them on the frame
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            confidence = confidences[i].item()  # Confidence score
            label_id = int(class_ids[i].item())  # Get class index (e.g., mask or no mask)
            label = labels[label_id]  # Get the label for the class

            if confidence > 0.5:  # Only consider detections with confidence > 50%
                if label == 'no mask':  # If the label is 'no mask', use red box
                    color = (0, 0, 255)  # Red (for no mask)
                    save_image(frame, index)  # Save image with no mask
                    index += 1
                elif label == 'mask':  # If the label is 'mask', use blue box
                    color = (255, 0, 0)  # Blue (for mask)

                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Live Video - Face Mask Detection', frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
