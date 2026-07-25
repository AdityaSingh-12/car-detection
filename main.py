from ultralytics import YOLO
import cv2
import time
from collections import deque, Counter
from collections import defaultdict

slot_map = {
    "Hatchback": "Small Slot",
    "Sedan": "Medium Slot",
    "SUV": "Large Slot",
    "Pick-Up": "XL Slot",
    "Van": "XL Slot",
    "Coupe": "Medium Slot",
    "Convertible": "Medium Slot",
    "Wagon": "Socha nhi mene"
}

# Detection model
detector = YOLO("model/yolo11n.pt")

# Classification model
classifier = YOLO("model/best.pt")

# Store last 10 predictions
prediction_history = deque(maxlen=10)

# Average confidence store karega
confidence_sum = defaultdict(float)

# Har class kitni baar aayi
confidence_count = defaultdict(int)

# Minimum confidence required
CONF_THRESHOLD = 0.70

# COCO class id for car
CAR_CLASS_ID = 2

cap = cv2.VideoCapture(0)

start_time = time.time()
SCAN_TIME = 10  # Camera kitne seconds chalega

if not cap.isOpened():
    print("Camera not opened!")
    exit()

while True:
    print("Loop Running")
    ret, frame = cap.read()

    if not ret:
        break

    results = detector(frame, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])

        # Sirf car detect hone par classify karenge
        if cls_id == CAR_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop car
            car_crop = frame[y1:y2, x1:x2]

            if car_crop.size == 0:
                continue

            # Classification
            cls_result = classifier(car_crop, verbose=False)

            class_id = cls_result[0].probs.top1
            class_name = cls_result[0].names[class_id]
            confidence = float(cls_result[0].probs.top1conf)

            # Only keep confident predictions
            if confidence >= 0.40:
                prediction_history.append(class_name)

            # Inhe 'if cls_id == CAR_CLASS_ID' block ke andar hona chahiye
            confidence_sum[class_name] += confidence
            confidence_count[class_name] += 1

            # Stable prediction
            if len(prediction_history) > 0:
                stable_class = Counter(prediction_history).most_common(1)[0][0]
            else:
                stable_class = "Unknown"

            slot = slot_map.get(stable_class, "Unknown Slot")

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put class name
            cv2.putText(
                frame,
                f"{stable_class} | {slot} | {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
    cv2.imshow("AI Parking", frame)
    
    # Stop scanning after 5 seconds
    if time.time() - start_time >= SCAN_TIME:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("\n============================")
print("AI PARKING ANALYSIS")
print("============================")

if len(confidence_sum) == 0:
    print("No vehicle detected.")
else:
    averages = {}
    for cls in confidence_sum:
        averages[cls] = confidence_sum[cls] / confidence_count[cls]

    sorted_avg = sorted(
        averages.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nPrediction Confidence\n")
    for cls, conf in sorted_avg[:3]:
        print(f"{cls:15} : {conf*100:.2f}%")

    final_vehicle = sorted_avg[0][0]

    print("\n----------------------------")
    print("Final Decision :", final_vehicle)
    print("Recommended Slot :", slot_map.get(final_vehicle, "Unknown Slot"))

print("============================")
cap.release()
cv2.destroyAllWindows()