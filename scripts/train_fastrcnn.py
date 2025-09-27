
import os
import glob
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths

# Updated paths for new project structure

# Updated paths for your actual data structure

import os

# Get absolute path to the project root (one level up from scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

train_img_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'train', 'images')
train_lbl_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'train', 'labels')
val_img_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'valid', 'images')
val_lbl_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'valid', 'labels')
model_save_path = os.path.join(PROJECT_ROOT, 'models', 'fastrcnn', 'final_model2.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Label mapping
class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
label2target = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

def getlabels(path):
    df = pd.read_csv(path, names=['label', 'x_center', 'y_center', 'width', 'height'], delimiter=' ')
    df['xmin'] = (df['x_center'] - df['width'] / 2).clip(lower=0.0)
    df['ymin'] = (df['y_center'] - df['height'] / 2).clip(lower=0.0)
    df['xmax'] = (df['x_center'] + df['width'] / 2).clip(upper=1.0)
    df['ymax'] = (df['y_center'] + df['height'] / 2).clip(upper=1.0)
    df = df.drop(columns=['x_center', 'y_center', 'width', 'height'])
    return df

def preprocess_data(img):
    img = torch.tensor(img).float().permute(2,0,1)
    img = img / 255.0
    return img

class OpenDataset(Dataset):
    def __init__(self, imgpaths, boxpaths):
        self.imgpaths = imgpaths
        self.boxpaths = boxpaths

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, ix):
        imgpath = self.imgpaths[ix]
        img = Image.open(imgpath).convert("RGB")
        img = img.resize((640, 640), resample=Image.BILINEAR)
        img = np.array(img)
        img = preprocess_data(img)
        boxpath = self.boxpaths[ix]
        data = getlabels(boxpath)
        if data.empty:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = data[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
            boxes[:, [0, 2]] *= 640
            boxes[:, [1, 3]] *= 640
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(data['label'].values, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}
        return img, target
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


def get_dataloader(img_dir, lbl_dir, batch_size=4, shuffle=True):
    # Map base name to full image path (handle multiple extensions)
    img_map = {}
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(f)[0]
            img_map[base] = os.path.join(img_dir, f)
    # Map base name to full label path
    lbl_map = {}
    for f in os.listdir(lbl_dir):
        if f.lower().endswith('.txt'):
            base = os.path.splitext(f)[0]
            lbl_map[base] = os.path.join(lbl_dir, f)
    # Only use pairs where both exist
    common_bases = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    imgpaths = []
    boxpaths = []
    missing_imgs = []
    missing_lbls = []
    for base in common_bases:
        if os.path.exists(img_map[base]) and os.path.exists(lbl_map[base]):
            imgpaths.append(img_map[base])
            boxpaths.append(lbl_map[base])
        else:
            if not os.path.exists(img_map[base]):
                missing_imgs.append(img_map[base])
            if not os.path.exists(lbl_map[base]):
                missing_lbls.append(lbl_map[base])
    print(f"Found {len(imgpaths)} valid image/label pairs in {img_dir} and {lbl_dir}")
    if missing_imgs:
        print(f"Warning: {len(missing_imgs)} label(s) had no matching image file.")
    if missing_lbls:
        print(f"Warning: {len(missing_lbls)} image(s) had no matching label file.")
    dataset = OpenDataset(imgpaths, boxpaths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, collate_fn=OpenDataset.collate_fn)

# DataLoaders

train_loader = get_dataloader(train_img_dir, train_lbl_dir)
val_loader = get_dataloader(val_img_dir, val_lbl_dir, shuffle=False)

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0
    for images, targets in tqdm(train_loader, desc="Training"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}")

# Save model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Save loss graphs
os.makedirs('outputs', exist_ok=True)
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('FastRCNN Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('outputs/fastrcnn_loss_curve.png')
plt.close()
print('Loss curve saved to outputs/fastrcnn_loss_curve.png')
