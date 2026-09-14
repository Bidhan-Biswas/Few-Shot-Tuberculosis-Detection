import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================
# SETTINGS
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MENDELEY_ROOT = r"D:\Research Work\TB Classification\Mendeley Dataset\Mendeley Data"
TB11K_MODEL_PATH = r"D:\Research Work\TB Classification\best_tb11k_binary_densenet.pth"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
FEW_SHOT_PER_CLASS = 75  # change here

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
# LOAD DATA
# ==============================
dataset = ImageFolder(root=MENDELEY_ROOT, transform=transform)
class_names = dataset.classes
print("Classes:", class_names)

# ==============================
# FEW SHOT SPLIT (Balanced)
# ==============================
targets = np.array(dataset.targets)

few_indices = []
test_indices = []

for c in range(len(class_names)):
    class_idx = np.where(targets == c)[0]
    np.random.shuffle(class_idx)

    few_indices.extend(class_idx[:FEW_SHOT_PER_CLASS])
    test_indices.extend(class_idx[FEW_SHOT_PER_CLASS:])

train_dataset = Subset(dataset, few_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Few-shot train size:", len(train_dataset))
print("Test size:", len(test_dataset))

# ==============================
# MODEL FACTORY
# ==============================
def create_model(pretrained_tb11k=True):
    model = models.densenet121(weights=None)

    if pretrained_tb11k:
        state_dict = torch.load(TB11K_MODEL_PATH, map_location=DEVICE)
        new_state_dict = {}
        for k,v in state_dict.items():
            name = k.replace("model.","")
            if "classifier" not in name:
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model.to(DEVICE)

# ==============================
# EVALUATION
# ==============================
def evaluate(model):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return acc, f1, auc

# ==============================
# TRAINING FUNCTION
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

# -------------------------------
# EXP 1: ZERO SHOT
# -------------------------------
print("\n==== EXP 1: Zero-Shot (No Training) ====")
model = create_model(pretrained_tb11k=True)
acc, f1, auc = evaluate(model)
print("Zero-shot : ACC:", acc, "F1:", f1, "AUC:", auc)

# -------------------------------
# EXP 2: FEW SHOT (Frozen)
# -------------------------------
print("\n==== EXP 2: Few-Shot Frozen Backbone ====")
model = create_model(pretrained_tb11k=True)
model = train_model(model, freeze_backbone=True)
acc, f1, auc = evaluate(model)
print("Few-shot Frozen : ACC:", acc, "F1:", f1, "AUC:", auc)

# -------------------------------
# EXP 3: FULL FINE-TUNE
# -------------------------------
print("\n==== EXP 3: Few-Shot Full Fine-Tune ====")
model = create_model(pretrained_tb11k=True)
model = train_model(model, freeze_backbone=False)
acc, f1, auc = evaluate(model)
print("Few-shot Full : ACC:", acc, "F1:", f1, "AUC:", auc)

# -------------------------------
# EXP 4: SCRATCH BASELINE
# -------------------------------
print("\n==== EXP 4: Scratch Training ====")
model = create_model(pretrained_tb11k=False)
model = train_model(model, freeze_backbone=False)
acc, f1, auc = evaluate(model)
print("Scratch : ACC:", acc, "F1:", f1, "AUC:", auc)