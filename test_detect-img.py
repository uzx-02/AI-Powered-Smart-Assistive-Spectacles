from ultralytics import YOLO
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define test image path
img_path = "img2.jpg"

# Perform object detection
results = model(img_path)

# Define ground truth manually (example: 1 person, 1 car)
ground_truth = [
    (0, 150, 100, 300, 400),  # Person (class 0)
    (2, 400, 250, 600, 450)   # Car (class 2)
]

# Extract predicted classes
predicted_classes = []
for result in results:
    for box in result.boxes:
        predicted_classes.append(int(box.cls[0]))  # Convert tensor to int

# Get ground truth class labels
gt_classes_list = [obj[0] for obj in ground_truth]

# Ensure lengths match
def match_predictions_with_ground_truth(gt_classes, pred_classes):
    matched_preds = []
    matched_gt = []
    for gt in gt_classes:
        if gt in pred_classes:
            matched_preds.append(gt)
            matched_gt.append(gt)
        else:
            matched_gt.append(gt)
            matched_preds.append(-1)  # -1 for no match
    return matched_gt, matched_preds

matched_gt, matched_preds = match_predictions_with_ground_truth(gt_classes_list, predicted_classes)

# Convert to tensors
gt_tensor = torch.tensor(matched_gt, dtype=torch.int64)
pred_tensor = torch.tensor(matched_preds, dtype=torch.int64)

# Compute evaluation metrics
if len(gt_tensor) > 0 and len(pred_tensor) > 0:
    precision = precision_score(gt_tensor.numpy(), pred_tensor.numpy(), average='macro', zero_division=0)
    recall = recall_score(gt_tensor.numpy(), pred_tensor.numpy(), average='macro', zero_division=0)
    f1 = f1_score(gt_tensor.numpy(), pred_tensor.numpy(), average='macro', zero_division=0)
else:
    precision, recall, f1 = 0, 0, 0

# Print evaluation results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
