import os
import cv2
import glob
from ultralytics import YOLO

weights_list = glob.glob("runs/detect/*/weights/best.pt")
if len(weights_list) == 0:
    raise FileNotFoundError(
        "No trained YOLOv8 model found. Please train first using train_smoke_model.py"
    )

model_path = weights_list[-1]
print("Using model:", model_path)

model = YOLO(model_path)
print("Trained YOLOv8 model loaded successfully!")

video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video file {video_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, (1500, 800))
    results = model(frame_small)
    annotated_frame = results[0].plot()
    cv2.imshow("Smoke Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()