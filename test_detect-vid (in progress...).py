import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# --------------------- CONFIGURATION ---------------------
MODEL_PATH = "yolov8n.pt"
INPUT_VIDEO = "vid2.mp4"   # Path to your input video file
OUTPUT_VIDEO = "test_output-vid.mp4"          # Path for the output annotated video

# Detection and confidence settings
CONFIDENCE_THRESHOLD = 0.5

# Camera and object parameters (assumed/calibrated for your setup)
CAMERA_RESOLUTION = (640, 480)  # (width, height)
FOCAL_LENGTH = 3.04            # in mm (example value)
SENSOR_WIDTH = 3.68            # in mm (example value)
px_per_mm = CAMERA_RESOLUTION[0] / SENSOR_WIDTH  # pixels per mm

# Known object heights in meters (for distance estimation)
KNOWN_HEIGHTS = {
    0: 1.7,   # person
    2: 1.5,   # car
    3: 2.5,   # motorcycle
    5: 2.0,   # bus
    7: 0.5    # truck
}

# Alert distance thresholds (in meters)
CRITICAL_DISTANCE = 1.5   # Critical: high alert
CAUTION_DISTANCE = 3.0    # Caution: normal caution

# Center path definition (consider objects roughly within ±20% of the frame center)
CENTER_TOLERANCE = CAMERA_RESOLUTION[0] * 0.2  # 20% of frame width

# --------------------- HELPER FUNCTIONS ---------------------
def is_in_center(x_center, frame_width):
    center = frame_width / 2
    return abs(x_center - center) <= CENTER_TOLERANCE

def estimate_distance(cls_id, bbox_height):
    """
    Estimate distance using a simple perspective projection formula.
    bbox_height: height of the bounding box in pixels.
    Returns distance in meters.
    """
    if cls_id not in KNOWN_HEIGHTS:
        return None
    object_height = KNOWN_HEIGHTS[cls_id]  # in meters
    image_height_mm = bbox_height / px_per_mm  # convert bbox height from pixels to mm
    if image_height_mm == 0:
        return None
    # distance (in meters) = (object height * focal length) / (image height in mm converted to meters)
    distance = (object_height * FOCAL_LENGTH) / (image_height_mm * 1e-3)
    return distance

# --------------------- MODEL INITIALIZATION ---------------------
model = YOLO(MODEL_PATH)

# --------------------- VIDEO I/O SETUP ---------------------
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# --------------------- PROCESS VIDEO ---------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the current frame
    results = model(frame)  # Model inference
    boxes = results[0].boxes  # Access detection boxes

    for box in boxes:
        if box.conf < CONFIDENCE_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        bbox = box.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = map(int, bbox)
        x_center = (x_min + x_max) / 2
        bbox_height = y_max - y_min

        # Estimate distance; if estimation fails, assume a far distance
        distance = estimate_distance(cls_id, bbox_height)
        if distance is None:
            distance = 100  # Arbitrarily high to mark as "normal"

        # Determine if detection is in the center path
        in_center = is_in_center(x_center, frame_width)

        # Decide on box color and tag based on alert level (only consider center detections for caution)
        if in_center:
            if distance <= CRITICAL_DISTANCE:
                color = (0, 0, 255)       # Red for high alert
                tag = "high alert"
            elif distance <= CAUTION_DISTANCE:
                color = (0, 165, 255)     # Orange for caution
                tag = "caution"
            else:
                color = (0, 255, 0)       # Green for normal
                tag = "normal"
        else:
            color = (0, 255, 0)
            tag = "normal"

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        label = f"{tag} ({distance:.1f}m)"
        cv2.putText(frame, label, (x_min, max(y_min - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Optionally, display the frame during processing for debugging (commented out for headless operation)
    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output video saved as", OUTPUT_VIDEO)
