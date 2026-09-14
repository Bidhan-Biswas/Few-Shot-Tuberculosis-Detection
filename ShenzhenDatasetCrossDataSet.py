import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)

import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# SETTINGS
# ==============================
SHENZHEN_ROOT = r"D:\Research Work\TB Classification\Shenzhen Dataset"
MODEL_PATH = r"D:\Research Work\TB Classification\best_tb11k_binary_densenet.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

print("Using device:", DEVICE)

# ==============================
# TRANSFORMS (IMPORTANT)
# ==============================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# SHENZHEN DATASET
# ==============================
class ShenzhenDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        csv_path = os.path.join(root, "shenzhen_metadata.csv")
        img_dir = os.path.join(root, "images")

        df = pd.read_csv(csv_path)

        self.samples = []

        for _, row in df.iterrows():
            img_name = str(row["study_id"]).strip()
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            findings = str(row["findings"]).lower()

            # Label from metadata
            label = 0 if "normal" in findings else 1

            self.samples.append((img_path, label))

        print(f"Loaded Shenzhen samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ==============================
# LOAD DATA
# ==============================
dataset = ShenzhenDataset(
    root=SHENZHEN_ROOT,
    transform=test_transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# LOAD MODEL (NO TRAINING)
# ==============================
model = models.densenet121(weights=None)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace("model.", "")
    if "classifier" not in name:
        new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)
model.classifier = nn.Linear(model.classifier.in_features, 2)

model.to(DEVICE)
model.eval()

print("Model loaded successfully")

# ==============================
# EVALUATION
# ==============================
all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = (probs > 0.5).int()   # threshold adjustable

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
acc = np.mean(all_preds == all_labels)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print("\n=== Shenzhen External Evaluation ===")
print("Accuracy:", acc)
print("F1 Score:", f1)
print("AUC:", auc)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ==============================
# ROC CURVE
# ==============================
fpr, tpr, _ = roc_curve(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Shenzhen External Test)")
plt.show()