import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    brier_score_loss
)

import numpy as np
import os
from collections import defaultdict
import random

# ==============================
# SETTINGS
# ==============================
MENDELEY_ROOT = r"D:\Research Work\TB Classification\Mendeley Dataset\Mendeley Data"
MODEL_PATH = r"D:\Research Work\TB Classification\best_tb11k_binary_densenet.pth"
SAVE_PATH = "adapted_mendeley_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-5
TRAIN_RATIO = 0.15

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Using device:", DEVICE)

# ==============================
# TRANSFORMS
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==============================
# LOAD DATASET
# ==============================
dataset = ImageFolder(root=MENDELEY_ROOT, transform=train_transform)
targets = np.array(dataset.targets)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

# ==============================
# GROUPING (proxy patient split)
# ==============================
def extract_group_id(filepath):
    """
    Attempt to extract patient/group ID from filename.
    MODIFY this based on your dataset naming pattern.
    """
    name = os.path.basename(filepath)

    # Example patterns:
    # patient123_scan1.png → patient123
    # IMG_00123.png → IMG_00123

    return name.split("_")[0]  # <-- customize if needed


group_dict = defaultdict(list)

for idx, (path, _) in enumerate(dataset.samples):
    gid = extract_group_id(path)
    group_dict[gid].append(idx)

groups = list(group_dict.keys())
random.shuffle(groups)

# ==============================
# GROUP-LEVEL SPLIT
# ==============================
train_indices = []
test_indices = []

for gid in groups:
    if len(train_indices) < TRAIN_RATIO * len(dataset):
        train_indices.extend(group_dict[gid])
    else:
        test_indices.extend(group_dict[gid])

print("\n=== GROUP SPLIT INFO ===")
print(f"Total groups: {len(groups)}")
print(f"Train images: {len(train_indices)}")
print(f"Test images : {len(test_indices)}")

# Verify no overlap
assert len(set(train_indices) & set(test_indices)) == 0, "Leakage detected!"

# Create subsets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Apply correct transform to test
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# MODEL
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

print("Model loaded (feature extractor + new classifier)")

optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ==============================
# TRAIN
# ==============================
def train_one_epoch():
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# ==============================
# EVALUATE
# ==============================
def evaluate():
    model.eval()

    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = np.mean(all_preds == all_labels)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    brier = brier_score_loss(all_labels, all_probs)

    print("\n=== METRICS ===")
    print(f"ACC  : {acc:.4f}")
    print(f"F1   : {f1:.4f}")
    print(f"AUC  : {auc:.4f}")
    print(f"Sens : {sensitivity:.4f}")
    print(f"Spec : {specificity:.4f}")
    print(f"Brier: {brier:.4f}")

    print("\nConfusion Matrix:\n", cm)

    return auc

# ==============================
# TRAIN LOOP
# ==============================
best_auc = 0

for epoch in range(EPOCHS):
    loss = train_one_epoch()
    print(f"\nEpoch {epoch+1}/{EPOCHS} Loss: {loss:.4f}")

    auc = evaluate()

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), SAVE_PATH)
        print("Best model saved!")

print("\nTraining done. Best AUC:", best_auc)