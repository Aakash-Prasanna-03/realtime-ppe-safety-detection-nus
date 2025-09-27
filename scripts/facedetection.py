import os
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
from collections import defaultdict

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known embeddings
with open(r'models/embeddings/face_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
known_embeddings = data['embeddings']
known_labels = data['labels']
registered_users = set(known_labels)

# Paths
image_folder = 'outputs/screenshots'
profile_folder = 'data/user_profiles'
violation_csv = 'outputs/reports/violations_report.csv'
os.makedirs(profile_folder, exist_ok=True)

# Load existing CSV (if any)
if os.path.exists(violation_csv):
    df_existing = pd.read_csv(violation_csv)
else:
    df_existing = pd.DataFrame(columns=["Emp ID", "Emp Name", "Pic URL", "Violations"])

# Create lookup for existing names
violations_map = df_existing.set_index("Emp Name").to_dict(orient="index")
existing_names = list(violations_map.keys())

# Temp storage
unknown_embeddings = []
unknown_names = []
session_counts = defaultdict(lambda: {"count": 0, "pic_path": ""})

# Process screenshots
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path).convert('RGB')
        boxes, _ = mtcnn.detect(img)
        if boxes is None:
            continue

        faces = mtcnn(img)
        for face in faces:
            if face is None:
                continue
            with torch.no_grad():
                embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()

            similarities = cosine_similarity(embedding, known_embeddings)[0]
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]

            if best_similarity > 0.75:
                name = known_labels[best_match_idx]
            else:
                # Check if same unknown already seen
                matched = False
                for i, u_embed in enumerate(unknown_embeddings):
                    sim = cosine_similarity(embedding, u_embed)[0][0]
                    if sim > 0.75:
                        name = unknown_names[i]
                        matched = True
                        break
                if not matched:
                    name = f"Unknown {len(unknown_names) + 1}"
                    unknown_embeddings.append(embedding)
                    unknown_names.append(name)

            session_counts[name]["count"] += 1
            if "Unknown" in name:
                session_counts[name]["pic_path"] = image_path

# Update existing or append new entries
for name, data in session_counts.items():
    if name in violations_map:
        prev = int(violations_map[name]["Violations"])
        violations_map[name]["Violations"] = prev + data["count"]
    else:
        emp_id = "N/A" if "Unknown" in name else f"E{len(df_existing) + 1:03}"
        pic_path = data["pic_path"] if "Unknown" in name else ""
        violations_map[name] = {
            "Emp ID": emp_id,
            "Pic URL": pic_path,
            "Violations": data["count"]
        }

# Convert back to DataFrame
rows = []
for name, info in violations_map.items():
    rows.append({
        "Emp ID": info["Emp ID"],
        "Emp Name": name,
        "Pic URL": info["Pic URL"],
        "Violations": info["Violations"]
    })

df_final = pd.DataFrame(rows)
df_final.to_csv(violation_csv, index=False)
print("✅ Violations updated and saved to:", violation_csv)

''' Clear screenshots folder
for file in os.listdir(image_folder):
    file_path = os.path.join(image_folder, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
print("🧹 Screenshots folder cleared.")'''
