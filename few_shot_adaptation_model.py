import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
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
MENDELEY_ROOT = r"Mendeley Dataset\Mendeley Data"
MODEL_PATH = "best_tb11k_binary_densenet.pth"
SAVE_PATH = "adapted_mendeley_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-5
TRAIN_SPLIT = 0.15

print("Using device:", DEVICE)

# ==============================
# TRANSFORMS
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# LOAD DATA
# ==============================
full_dataset = ImageFolder(root=MENDELEY_ROOT,
                           transform=train_transform)

train_size = int(TRAIN_SPLIT * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset,
                                           [train_size, test_size])

test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

print("Classes:", full_dataset.classes)
print("Train images:", len(train_dataset))
print("Test images:", len(test_dataset))

# Create plain DenseNet121
model = models.densenet121(weights=None)

# Load the saved state_dict (with 'model.' prefix)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Create a new state_dict without the old classifier keys
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace("model.", "")  # remove prefix
    if "classifier" not in name:    # skip old classifier keys
        new_state_dict[name] = v

# Load only the feature extractor parts
model.load_state_dict(new_state_dict, strict=False)  # strict=False ignores missing classifier keys

# Now replace classifier with your new single Linear layer
model.classifier = nn.Linear(model.classifier.in_features, 2)

model.to(DEVICE)

print("Loaded features successfully (old classifier ignored, new one attached)")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

# ==============================
# TRAIN FUNCTION
# ==============================
def train_one_epoch():
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# ==============================
# EVALUATION FUNCTION
# ==============================
def evaluate():
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = np.mean(all_preds == all_labels)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print("\nAccuracy:", acc)
    print("F1 Score:", f1)
    print("AUC:", auc)

    print("\nClassification Report:")
    print(classification_report(all_labels,
                                all_preds,
                                target_names=full_dataset.classes))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return auc

# ==============================
# TRAINING LOOP
# ==============================
best_auc = 0

print("\nStarting Fine-Tuning...\n")

for epoch in range(EPOCHS):
    loss = train_one_epoch()
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f}")

    auc = evaluate()

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), SAVE_PATH)
        print("Best model saved!")

print("\nTraining Finished")
print("Best AUC:", best_auc)

# ==============================
# ROC CURVE
# ==============================
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:,1]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

fpr, tpr, _ = roc_curve(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()