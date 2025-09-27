import cv2
import torch
import numpy as np
import time
from collections import defaultdict
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

model = fasterrcnn_resnet50_fpn(weights=None)
num_classes = 10
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model_path = r"models/fastrcnn/final_model2.pth"
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

class_dict = {
    0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask',
    4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone',
    7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'
}
unsafe_classes = {'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}
mask = {'NO-Mask'}

def preprocess_frame(frame):
    resized = cv2.resize(frame, (512, 512))
    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img).unsqueeze(0).to(device)
    return img_tensor, resized

def decode_output(output, thresh=0.5):
    boxes = output['boxes'].detach().cpu().numpy()
    scores = output['scores'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()
    keep = scores >= thresh
    return boxes[keep], scores[keep], labels[keep]

def draw_boxes(frame, boxes, labels, scores):
    if len(boxes) == 0:
        cv2.rectangle(frame, (50, 50), (300, 120), (0, 255, 255), -1)
        cv2.putText(frame, "Nothing Detected", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            label_text = class_dict.get(label, f"Class {label}")
            color = (0, 0, 255) if label_text in unsafe_classes else (0, 255, 0)
            text = f"{label_text} @{score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ✅ Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

# new logic vars
no_mask_start_time = None
screenshot_taken = False
continuous_no_mask_duration = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_tensor, resized_frame = preprocess_frame(frame)

    with torch.no_grad():
        output = model(img_tensor)[0]
        boxes, scores, labels = decode_output(output, thresh=0.5)

    current_time = time.time()
    no_mask_detected = any(class_dict.get(label, "") in mask for label in labels)

    if no_mask_detected:
        if no_mask_start_time is None:
            no_mask_start_time = current_time
        elif (current_time - no_mask_start_time >= continuous_no_mask_duration) and not screenshot_taken:
            timestamp = time.strftime("%Y-%m%d-%H-%M-%S")
            filename = rf"outputs/screenshots/violation_{timestamp}.png"
            cv2.imwrite(filename, resized_frame)
            print(f"📸 Screenshot saved: {filename}")
            no_mask_start_time = current_time

    else:
        # reset timer and flag if no no-mask anymore
        no_mask_start_time = None
        screenshot_taken = False

    frame_with_boxes = draw_boxes(resized_frame, boxes, labels, scores)
    cv2.imshow("🔍 Real-Time Safety Monitor", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
