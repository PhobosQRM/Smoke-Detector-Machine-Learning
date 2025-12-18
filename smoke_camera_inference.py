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

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    raise FileNotFoundError("Cannot open webcam. Please check if it is connected.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    frame_small = cv2.resize(frame, (640, 480))

    results = model(frame_small)

    annotated_frame = results[0].plot()

    cv2.imshow("Smoke Detection (PC Webcam)", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()