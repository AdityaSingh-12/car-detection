import time
from collections import Counter, defaultdict, deque
import cv2
from ultralytics import YOLO

# Slot mapping
slot_map = {
    "Hatchback": "Small Slot",
    "Sedan": "Medium Slot",
    "SUV": "Large Slot",
    "Pick-Up": "XL Slot",
    "Van": "XL Slot",
    "Coupe": "Medium Slot",
    "Convertible": "Medium Slot",
    "Wagon": "Medium Slot",
}

class_options = {
    "1": "Hatchback",
    "2": "Sedan",
    "3": "SUV",
    "4": "Pick-Up",
    "5": "Van",
    "6": "Coupe",
    "7": "Convertible",
    "8": "Wagon",
}

# Detection model
detector = YOLO("model/yolo11n.pt")
# Classification model
classifier = YOLO("model/best.pt")

# Store recent predictions for stabilization
prediction_history = deque(maxlen=20)

# Confidence tracking
confidence_sum = defaultdict(float)
confidence_count = defaultdict(int)

# Configurations
CONF_THRESHOLD = 0.40  # Minimum classification confidence
CAR_CLASS_ID = 2       # COCO class ID for car
SCAN_TIME = 10         # Camera scan duration in seconds

cap = cv2.VideoCapture(0)
start_time = time.time()

if not cap.isOpened():
    print("Error: Camera not opened!")
    exit()

print("Scanning started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame, verbose=False)

    if results and results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            det_conf = float(box.conf[0])

            # Process only cars detected with sufficient detection confidence
            if cls_id == CAR_CLASS_ID and det_conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop vehicle region safely
                car_crop = frame[y1:y2, x1:x2]
                if car_crop.size == 0:
                    continue

                # Classification
                cls_result = classifier(car_crop, verbose=False)
                class_id = cls_result[0].probs.top1
                class_name = cls_result[0].names[class_id]
                confidence = float(cls_result[0].probs.top1conf)

                # Keep only confident predictions
                if confidence >= CONF_THRESHOLD:
                    prediction_history.append(class_name)
                    confidence_sum[class_name] += confidence
                    confidence_count[class_name] += 1

                # Calculate stable prediction using mode of history
                if len(prediction_history) > 0:
                    stable_class = Counter(prediction_history).most_common(1)[0][0]
                else:
                    stable_class = "Unknown"

                slot = slot_map.get(stable_class, "Unknown Slot")

                # Draw bounding box and display info
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{stable_class} | {slot} | {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

    cv2.imshow("AI Parking", frame)

    # Stop scanning after set timeout or keypress 'q'
    if (time.time() - start_time >= SCAN_TIME) or (cv2.waitKey(1) & 0xFF == ord("q")):
        break

# Release camera and close GUI windows before terminal interaction
cap.release()
cv2.destroyAllWindows()

# Final Summary Report
print("\n============================")
print("AI PARKING ANALYSIS")
print("============================")

if len(confidence_sum) == 0:
    print("No vehicle detected during scan time.")
else:
    averages = {}
    for cls in confidence_sum:
        averages[cls] = confidence_sum[cls] / confidence_count[cls]

    sorted_avg = sorted(
        averages.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nPrediction Confidence:\n")
    for cls, conf in sorted_avg[:3]:
        print(f"{cls:15} : {conf * 100:.2f}%")

    final_vehicle = sorted_avg[0][0]
    print("\nModel Prediction:", final_vehicle)

    # User confirmation & correction logic
    correct = input("Is prediction correct? (y/n): ")
    if correct.lower() == "n":
        print("\nSelect correct vehicle type:")
        for key, value in class_options.items():
            print(f"{key} : {value}")
        choice = input("Enter choice number: ")
        if choice in class_options:
            final_vehicle = class_options[choice]
            print("Corrected vehicle class:", final_vehicle)
        else:
            print("Invalid choice! Keeping original model prediction.")

    # Print decision regardless of choice
    print("\n----------------------------")
    print("Final Decision   :", final_vehicle)
    print("Recommended Slot :", slot_map.get(final_vehicle, "Unknown Slot"))

print("============================")