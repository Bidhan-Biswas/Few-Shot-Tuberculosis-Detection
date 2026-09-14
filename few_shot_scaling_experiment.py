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
MENDELEY_ROOT = r"Mendeley Dataset\Mendeley Data"
TB11K_MODEL_PATH = "best_tb11k_binary_densenet.pth"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 8
SHOT_LIST = [10, 25, 50, 75, 100]

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
targets = np.array(dataset.targets)

print("Classes:", class_names)

# ==============================
# MODEL FACTORY
# ==============================
def create_model(pretrained_tb11k=True):
    model = models.densenet121(weights=None)

    if pretrained_tb11k:
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
# EVALUATION
# ==============================
def evaluate(model, test_loader):
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
# TRAIN FUNCTION
# ==============================
def train_model(model, train_loader, freeze_backbone=False):

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

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

        print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    return model

# =========================================================
# ================= FEW-SHOT LOOP =========================
# =========================================================

results = []

for shots in SHOT_LIST:

    print("\n================================================")
    print(f"SHOTS PER CLASS: {shots}")
    print("================================================")

    # Balanced Few-Shot Split
    few_indices = []
    test_indices = []

    for c in range(len(class_names)):
        class_idx = np.where(targets == c)[0]
        np.random.shuffle(class_idx)

        few_indices.extend(class_idx[:shots])
        test_indices.extend(class_idx[shots:])

    train_dataset = Subset(dataset, few_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    # ---------------- ZERO SHOT ----------------
    model = create_model(pretrained_tb11k=True)
    acc, f1, auc = evaluate(model, test_loader)
    print("Zero-shot: ACC:", acc, "AUC:", auc)

    # ---------------- FROZEN ----------------
    model = create_model(pretrained_tb11k=True)
    model = train_model(model, train_loader, freeze_backbone=True)
    acc_frozen, f1_frozen, auc_frozen = evaluate(model, test_loader)
    print("Frozen: ACC:", acc_frozen, "AUC:", auc_frozen)

    # ---------------- FULL ----------------
    model = create_model(pretrained_tb11k=True)
    model = train_model(model, train_loader, freeze_backbone=False)
    acc_full, f1_full, auc_full = evaluate(model, test_loader)
    print("Full: ACC:", acc_full, "AUC:", auc_full)

    # ---------------- SCRATCH ----------------
    model = create_model(pretrained_tb11k=False)
    model = train_model(model, train_loader, freeze_backbone=False)
    acc_scratch, f1_scratch, auc_scratch = evaluate(model, test_loader)
    print("Scratch: ACC:", acc_scratch, "AUC:", auc_scratch)

    results.append([shots,
                    acc_frozen, acc_full, acc_scratch])

print("\n\n===== FINAL SUMMARY =====")
for r in results:
    print(f"Shots: {r[0]} | Frozen: {r[1]:.4f} | Full: {r[2]:.4f} | Scratch: {r[3]:.4f}")