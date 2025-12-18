import os
from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="eouYxxPQN31kmxz2p29g")
project = rf.workspace("kenneth-w8rl3").project("smoke-uvylj-ahwql")
version = project.version(2)
dataset = version.download("yolov8")
                
print("Dataset downloaded to:", dataset.location)

weights_folder = "runs/train/exp/weights"
os.makedirs(weights_folder, exist_ok=True)
model_path = os.path.join(weights_folder, "best.pt")

if not os.path.exists(model_path):
    print("Training YOLOv8 model...")
    model = YOLO("yolov8n.yaml")

    model.train(
        data=dataset.location + "/data.yaml",
        epochs=100,      # Adjust for faster or longer training
        imgsz=320,      # Smaller image size for speed
        batch=8,
        augment=True
    )
    print(f"Training finished. Model saved to {weights_folder}/best.pt")
else:
    print("Model already exists. Skipping training.")