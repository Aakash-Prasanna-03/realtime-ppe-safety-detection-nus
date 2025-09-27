import cv2
import torch
import numpy as np
import time
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Load model
model = fasterrcnn_resnet50_fpn(weights=None)
num_classes = 10
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(r"models/fastrcnn/final_model2.pth", map_location=device))
model.to(device).eval()

class_dict = {
    0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask',
    4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone',
    7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'
}

mask_label = 3  # 'NO-Mask'
threshold = 0.5
screenshot_dir = r"outputs/screenshots"
screenshot_interval = 5  # seconds

# Timer storage for NO-MASK individuals
no_mask_timers = {}
screenshot_flags = {}

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
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        label_text = class_dict.get(label, f"Class {label}")
        color = (0, 0, 255) if label_text.startswith("NO-") else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label_text} @{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    img_tensor, resized_frame = preprocess_frame(frame)

    with torch.no_grad():
        output = model(img_tensor)[0]
        boxes, scores, labels = decode_output(output, threshold)

    frame_with_boxes = resized_frame.copy()
    frame_with_boxes = draw_boxes(frame_with_boxes, boxes, labels, scores)

    active_ids = set()

    for idx, (box, label) in enumerate(zip(boxes, labels)):
        if label == mask_label:
            x1, y1, x2, y2 = map(int, box)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            found_id = None

            # Find matching previous box
            for pid, prev_center in no_mask_timers.items():
                px, py = prev_center['center']
                if abs(px - center[0]) < 50 and abs(py - center[1]) < 50:
                    found_id = pid
                    break

            if found_id is None:
                found_id = len(no_mask_timers) + 1
                no_mask_timers[found_id] = {'start_time': current_time, 'center': center}
                screenshot_flags[found_id] = False
            else:
                no_mask_timers[found_id]['center'] = center

            active_ids.add(found_id)

            # Screenshot logic
            elapsed = current_time - no_mask_timers[found_id]['start_time']
            if elapsed >= screenshot_interval and not screenshot_flags[found_id]:
                crop = resized_frame[y1:y2, x1:x2]
                timestamp = time.strftime("%Y-%m%d-%H-%M-%S")
                filename = f"{screenshot_dir}/violation_{found_id}_{timestamp}.png"
                cv2.imwrite(filename, crop)
                print(f"📸 Screenshot taken for person {found_id}")
                screenshot_flags[found_id] = True

    # Clean up people no longer visible
    to_remove = []
    for pid in no_mask_timers:
        if pid not in active_ids:
            to_remove.append(pid)

    for pid in to_remove:
        del no_mask_timers[pid]
        del screenshot_flags[pid]

    cv2.imshow("🛡️ Safety Monitor", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
