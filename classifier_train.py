from ultralytics import YOLO
# Load pretrained classification model
model = YOLO("yolo11n-cls.pt")
# Train the model
model.train(
    data="dataset",
    epochs=70,
    imgsz=224,
    batch=8,    # CPU ke liye 8 better hai
    workers=2,  # CPU load kam rahega
    device="cpu"
)