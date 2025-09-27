# Detailed FastRCNN evaluation script
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay

# --- CONFIG ---
fastrcnn_model_path = os.path.join('models', 'fastrcnn', 'final_model2.pth')
val_img_dir = os.path.join('data', 'dataset', 'valid', 'images')
val_lbl_dir = os.path.join('data', 'dataset', 'valid', 'labels')
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
num_classes = len(class_names)

def getlabels(path):
    arr = np.loadtxt(path)
    if arr.size == 0:
        return None, None
    if arr.ndim == 1:
        arr = arr[None, :]
    labels = arr[:, 0].astype(int)
    boxes = arr[:, 1:5]
    # Convert from YOLO (cx,cy,w,h) to (xmin,ymin,xmax,ymax) in pixel coords
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * 640
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * 640
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2]/2) * 640
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3]/2) * 640
    return boxes_xyxy, labels

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((640, 640))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2,0,1)
    return img

def iou(boxA, boxB):
    # Compute IoU between two boxes (xyxy)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def match_predictions(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=0.5):
    # For each pred, find best matching gt (if any)
    matches = []
    used = set()
    for i, (pb, pl, ps) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        best_iou = 0
        best_j = -1
        for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if j in used or pl != gl:
                continue
            iou_val = iou(pb, gb)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_iou >= iou_thresh and best_j != -1:
            matches.append((i, best_j))
            used.add(best_j)
    return matches

def evaluate_fastrcnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(fastrcnn_model_path, map_location=device))
    model.to(device)
    model.eval()


    matched_gt_labels = []
    matched_pred_labels = []
    all_pred_scores = []
    per_class_tp = [0]*num_classes
    per_class_fp = [0]*num_classes
    per_class_fn = [0]*num_classes
    unmatched_pred = 0
    unmatched_gt = 0

    img_files = sorted([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for fname in tqdm(img_files, desc='FastRCNN Eval'):
        img_path = os.path.join(val_img_dir, fname)
        lbl_path = os.path.join(val_lbl_dir, os.path.splitext(fname)[0] + '.txt')
        if not os.path.exists(lbl_path):
            continue
        gt_boxes, gt_labels = getlabels(lbl_path)
        if gt_boxes is None or gt_labels is None:
            continue
        img = preprocess_image(img_path).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)[0]
        pred_boxes = output['boxes'].cpu().numpy()
        pred_scores = output['scores'].cpu().numpy()
        pred_labels = output['labels'].cpu().numpy()

        # --- DEBUG: Print Hardhat detections for images with Hardhat ground truth ---
        if 0 in gt_labels:
            print(f'Image: {fname} has Hardhat ground truth.')
            hardhat_preds = [(pb, ps) for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores) if pl == 0]
            if hardhat_preds:
                print(f'Predicted Hardhat detections: {len(hardhat_preds)}')
                for idx, (box, score) in enumerate(hardhat_preds):
                    print(f'  Box: {box}, Score: {score:.3f}')
            else:
                print('No Hardhat detections predicted.')

        matches = match_predictions(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
        matched_pred = set(i for i, _ in matches)
        matched_gt = set(j for _, j in matches)

        # For confusion matrix and PR: only use matched pairs
        for i, j in matches:
            matched_pred_labels.append(pred_labels[i])
            matched_gt_labels.append(gt_labels[j])
            all_pred_scores.append(pred_scores[i])

        # Per-class stats
        for c in range(num_classes):
            tp = sum(1 for i, j in matches if pred_labels[i]==c and gt_labels[j]==c)
            fp = sum(1 for i in range(len(pred_labels)) if pred_labels[i]==c and i not in matched_pred)
            fn = sum(1 for j in range(len(gt_labels)) if gt_labels[j]==c and j not in matched_gt)
            per_class_tp[c] += tp
            per_class_fp[c] += fp
            per_class_fn[c] += fn
        unmatched_pred += len(pred_labels) - len(matched_pred)
        unmatched_gt += len(gt_labels) - len(matched_gt)

    # Confusion Matrix (only matched pairs)
    if matched_gt_labels and matched_pred_labels:
        cm = confusion_matrix(matched_gt_labels, matched_pred_labels, labels=list(range(num_classes)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(10,8))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title('FastRCNN Confusion Matrix (matched pairs)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fastrcnn_confusion_matrix.png'))
        plt.close()
    else:
        print('No matched pairs for confusion matrix.')


    # Per-class Precision/Recall/F1/AP
    precisions = []
    recalls = []
    aps = []
    f1s = []
    for c in range(num_classes):
        tp = per_class_tp[c]
        fp = per_class_fp[c]
        fn = per_class_fn[c]
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        # AP: use sklearn average_precision_score if possible
        y_true = [1 if l==c else 0 for l in matched_gt_labels]
        y_score = [s if l==c else 0 for l,s in zip(matched_pred_labels, all_pred_scores)]
        try:
            ap = average_precision_score(y_true, y_score)
        except:
            ap = 0.0
        aps.append(ap)

    # Plot per-class PR
    x = np.arange(num_classes)
    plt.figure(figsize=(10,6))
    plt.bar(x-0.2, precisions, 0.4, label='Precision')
    plt.bar(x+0.2, recalls, 0.4, label='Recall')

    # Plot F1, Precision, Recall curves
    plt.figure(figsize=(10,6))
    plt.plot(x, f1s, marker='o', label='F1 Score')
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0,1)
    plt.title('FastRCNN Per-Class F1 Score Curve')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fastrcnn_f1_curve.png'))
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(x, precisions, marker='o', color='blue', label='Precision')
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0,1)
    plt.title('FastRCNN Per-Class Precision Curve')
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fastrcnn_p_curve.png'))
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(x, recalls, marker='o', color='green', label='Recall')
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0,1)
    plt.title('FastRCNN Per-Class Recall Curve')
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fastrcnn_r_curve.png'))
    plt.close()

    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0,1)
    plt.title('FastRCNN Per-Class Precision/Recall')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fastrcnn_per_class_pr.png'))
    plt.close()

    # Plot AP
    plt.figure(figsize=(10,6))
    plt.bar(x, aps)
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0,1)
    plt.title('FastRCNN Per-Class Average Precision (AP)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fastrcnn_per_class_ap.png'))
    plt.close()

    # Save results as CSV and Markdown
    import pandas as pd
    df = pd.DataFrame({
        'Class': class_names,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1s,
        'AP': aps
    })
    df.to_csv(os.path.join(output_dir, 'fastrcnn_per_class_metrics.csv'), index=False)
    md_table = '| Class | Precision | Recall | F1 | AP |\n|---|---|---|---|---|\n'
    for i in range(num_classes):
        md_table += f'| {class_names[i]} | {precisions[i]:.3f} | {recalls[i]:.3f} | {f1s[i]:.3f} | {aps[i]:.3f} |\n'
    md_table += f'| **mAP** |  |  |  | **{np.mean(aps):.3f}** |\n'
    with open(os.path.join(output_dir, 'fastrcnn_per_class_metrics.md'), 'w') as f:
        f.write(md_table)

    # Print summary for README
    print('\n===== FastRCNN Validation Results =====')
    print(md_table)
    print(f"mAP@0.5: {np.mean(aps):.3f}")
    print('Confusion matrix, PR, AP, and metrics table saved to outputs/.')

if __name__ == '__main__':
    evaluate_fastrcnn()
