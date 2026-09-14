import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ==============================
#  DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
#  PATHS (EDIT THESE)
# ==============================
CSV_PATH = r"TB11K\data.csv"
IMAGE_DIR = r"TB11K\images"

MODEL_SAVE_PATH = "best_tb11k_binary_densenet.pth"

# ==============================
#  DOMAIN-ROBUST AUGMENTATION
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
#  CUSTOM DATASET (CSV BASED)
# ==============================
class TB11KDataset(Dataset):
    def __init__(self, csv_path, image_dir, split="train", transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["source"] == split].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.image_dir, row["fname"])
        image = Image.open(img_path).convert("RGB")

        # Convert label
        if str(row["target"]).lower() == "no_tb":
            label = 0
        else:
            label = 1

        if self.transform:
            image = self.transform(image)

        return image, label

# ==============================
#  LOAD DATA
# ==============================
train_dataset = TB11KDataset(CSV_PATH, IMAGE_DIR, split="train", transform=train_transform)
val_dataset   = TB11KDataset(CSV_PATH, IMAGE_DIR, split="val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# ==============================
#  COMPUTE CLASS WEIGHTS
# ==============================
labels = []
for _, label in train_dataset:
    labels.append(label)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class Weights:", class_weights)

# ==============================
#  MODEL (Modified DenseNet)
# ==============================
class DenseNetBinary(nn.Module):
    def __init__(self):
        super(DenseNetBinary, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True)

        num_features = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.model(x)

model = DenseNetBinary().to(device)

# ==============================
#  LOSS + OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=3,
    factor=0.3
)

# ==============================
#  VALIDATION FUNCTION
# ==============================
def validate(model):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return val_loss / len(val_loader), acc

# ==============================
# TRAINING LOOP
# ==============================
def train_model(epochs=20):
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        loop = tqdm(train_loader)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        val_loss, val_acc = validate(model)
        scheduler.step(val_loss)

        print("\nTrain Loss:", running_loss / len(train_loader))
        print("Val Loss:", val_loss)
        print("Val Accuracy:", val_acc)
        print("-" * 40)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Best model saved!")

    print("Training Complete!")
    print("Best Validation Accuracy:", best_val_acc)

# ==============================
#  START TRAINING
# ==============================
if __name__ == "__main__":
    train_model(epochs=20) 