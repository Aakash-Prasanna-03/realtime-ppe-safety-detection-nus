import os

# Set your data directories here

import os

# Get absolute path to the project root (one level up from scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

train_img_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'train', 'images')
train_lbl_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'train', 'labels')
val_img_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'valid', 'images')
val_lbl_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'valid', 'labels')

def check_pairs(img_dir, lbl_dir):
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')])
    img_bases = set(os.path.splitext(f)[0] for f in img_files)
    lbl_bases = set(os.path.splitext(f)[0] for f in lbl_files)
    missing_labels = img_bases - lbl_bases
    missing_images = lbl_bases - img_bases
    print(f"Total images: {len(img_files)}")
    print(f"Total labels: {len(lbl_files)}")
    print(f"Images without labels: {len(missing_labels)}")
    if missing_labels:
        print("Images missing labels:")
        for f in sorted(missing_labels):
            print(f"  {f}")
    print(f"Labels without images: {len(missing_images)}")
    if missing_images:
        print("Labels missing images:")
        for f in sorted(missing_images):
            print(f"  {f}")
    print("---")

if __name__ == "__main__":
    print("Checking training set:")
    check_pairs(train_img_dir, train_lbl_dir)
    print("Checking validation set:")
    check_pairs(val_img_dir, val_lbl_dir)
