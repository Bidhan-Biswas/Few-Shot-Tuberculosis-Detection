import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import numpy as np
import random
import os

# ==============================
# REPRODUCIBILITY
# ==============================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ==============================
# SETTINGS
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MENDELEY_ROOT = r"D:\Research Work\TB Classification\Mendeley Dataset\Mendeley Data"
SHENZHEN_ROOT = r"D:\Research Work\TB Classification\Shenzhen Dataset"
TB11K_MODEL_PATH = r"D:\Research Work\TB Classification\best_tb11k_binary_densenet.pth"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
FEW_SHOT_PER_CLASS = 75

print("Using device:", DEVICE)

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
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
            img_name = str(row["study_id"])
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            findings = str(row["findings"]).lower()

            label = 0 if findings.strip() == "normal" else 1

            self.samples.append((img_path, label))

        print("Loaded Shenzhen samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# ==============================
# LOAD DATA
# ==============================
dataset = ImageFolder(root=MENDELEY_ROOT, transform=transform)
targets = np.array(dataset.targets)

# FEW SHOT SPLIT (Mendeley ONLY)
few_indices = []
for c in range(len(dataset.classes)):
    idxs = np.where(targets == c)[0]
    np.random.shuffle(idxs)
    few_indices.extend(idxs[:FEW_SHOT_PER_CLASS])

train_dataset = Subset(dataset, few_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Few-shot train size:", len(train_dataset))

# Shenzhen TEST ONLY
shenzhen_dataset = ShenzhenDataset(SHENZHEN_ROOT, transform=transform)
test_loader = DataLoader(shenzhen_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# MODEL
# ==============================
def create_model(pretrained=True):
    model = models.densenet121(weights=None)

    if pretrained:
        state_dict = torch.load(TB11K_MODEL_PATH, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("model.", "")
            if "classifier" not in name:
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model.to(DEVICE)

# ==============================
# EVALUATION (ON SHENZHEN)
# ==============================
def evaluate(model):
    model.eval()

    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:,1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ---- FIND BEST THRESHOLD ----
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thresh = thresholds[best_idx]

    preds = (all_probs >= best_thresh).astype(int)

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Best Threshold: {best_thresh:.4f}")
    return acc, f1, auc

# ==============================
# TRAIN FUNCTION
# ==============================
def train_model(model, freeze_backbone=False):

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
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

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    return model

# =====================================================
# ================= EXPERIMENTS =======================
# =====================================================

# ZERO-SHOT
print("\n==== EXP 1: Zero-Shot (TBX11K → Shenzhen) ====")
model = create_model(pretrained=True)
acc, f1, auc = evaluate(model)
print("ACC:", acc, "F1:", f1, "AUC:", auc)

# FEW-SHOT FROZEN
print("\n==== EXP 2: Few-Shot Frozen (Mendeley → Shenzhen) ====")
model = create_model(pretrained=True)
model = train_model(model, freeze_backbone=True)
acc, f1, auc = evaluate(model)
print("ACC:", acc, "F1:", f1, "AUC:", auc)

# FULL FINE-TUNE
print("\n==== EXP 3: Full Fine-Tune (Mendeley → Shenzhen) ====")
model = create_model(pretrained=True)
model = train_model(model, freeze_backbone=False)
acc, f1, auc = evaluate(model)
print("ACC:", acc, "F1:", f1, "AUC:", auc)

# SCRATCH
print("\n==== EXP 4: Scratch (Mendeley → Shenzhen) ====")
model = create_model(pretrained=False)
model = train_model(model, freeze_backbone=False)
acc, f1, auc = evaluate(model)
print("ACC:", acc, "F1:", f1, "AUC:", auc)