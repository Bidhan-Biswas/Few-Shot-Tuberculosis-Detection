import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report,
                             confusion_matrix)
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
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MENDELEY_ROOT   = r"D:\Research Work\TB Classification\Mendeley Dataset\Mendeley Data"
TB11K_MODEL_PATH = r"D:\Research Work\TB Classification\best_tb11k_binary_densenet.pth"

BATCH_SIZE  = 16
LR          = 1e-4
EPOCHS      = 8
SHOT_LIST   = [10, 25, 50, 75, 100]

# 20% of each class goes into the adaptation pool for k-shot sampling.
# The remaining 80% becomes the FIXED test set used for ALL experiments.
ADAPT_RATIO = 0.20

print("Using device:", DEVICE)

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# LOAD FULL DATASET
# ==============================
dataset     = ImageFolder(root=MENDELEY_ROOT, transform=transform)
class_names = dataset.classes
targets     = np.array(dataset.targets)

print("Classes:", class_names)
print("Class-to-index mapping:", dataset.class_to_idx)
print("Total images:", len(dataset))
for c, name in enumerate(class_names):
    print(f"  {name}: {(targets == c).sum()} images")

# ==============================
# PRE-SPLIT MENDELEY ONCE
# ── Done BEFORE the shot loop ──
# 20% → adaptation pool  (k-shot sampling happens here only)
# 80% → fixed test set   (same for ALL shot levels — fair comparison)
# ==============================
rng = np.random.default_rng(SEED)   # isolated RNG so main seed is not consumed

adapt_indices      = []
test_indices_fixed = []

for c in range(len(class_names)):
    class_idx = np.where(targets == c)[0].copy()
    rng.shuffle(class_idx)                          # reproducible shuffle

    n_adapt = int(len(class_idx) * ADAPT_RATIO)    # 20% for adaptation
    adapt_indices.extend(class_idx[:n_adapt].tolist())
    test_indices_fixed.extend(class_idx[n_adapt:].tolist())

# Build the fixed test loader — created ONCE, reused for every shot level
fixed_test_dataset = Subset(dataset, test_indices_fixed)
fixed_test_loader  = DataLoader(fixed_test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

# Report the fixed split composition
adapt_targets = targets[adapt_indices]
test_targets  = targets[test_indices_fixed]

print("\n=== FIXED DATA SPLIT (performed once before all experiments) ===")
for c, name in enumerate(class_names):
    n_adapt_c = int((adapt_targets == c).sum())
    n_test_c  = int((test_targets  == c).sum())
    print(f"  {name}:  {n_adapt_c} adapt  |  {n_test_c} test")
print(f"  Total adapt pool : {len(adapt_indices)} images")
print(f"  Total fixed test : {len(test_indices_fixed)} images")
print(f"  (Test set is IDENTICAL for k = {SHOT_LIST})")
print("=================================================================\n")

# ==============================
# MODEL FACTORY
# ==============================
def create_model(pretrained_tb11k=True):
    """
    Build DenseNet-121.
    If pretrained_tb11k=True  → load TBX11K-pretrained backbone weights.
    If pretrained_tb11k=False → random initialisation (scratch baseline).
    Classifier head is always replaced with a fresh 2-class linear layer.
    """
    model = models.densenet121(weights=None)

    if pretrained_tb11k:
        state_dict = torch.load(TB11K_MODEL_PATH, map_location=DEVICE)
        # Strip 'model.' prefix if present (from source training wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("model.", "")
            if "classifier" not in name:        # skip old classifier weights
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    # Replace classifier with a fresh 2-class head
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model.to(DEVICE)

# ==============================
# EVALUATION
# ==============================
def evaluate(model, loader, class_names=None):
    """
    Returns accuracy, F1, AUC and prints a full classification report
    including precision, recall, specificity, and confusion matrix.
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            preds   = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    # Full report
    target_names = class_names if class_names else ["Class 0", "Class 1"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=target_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(f"  TN (Normal correct): {cm[0,0]}  FP (Normal → TB): {cm[0,1]}")
    print(f"  FN (TB missed):      {cm[1,0]}  TP (TB correct):  {cm[1,1]}")

    # Derive specificity from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\n  Sensitivity (TB Recall): {sensitivity:.4f}")
    print(f"  Specificity (Normal Recall): {specificity:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

    return acc, f1, auc

# ==============================
# TRAINING FUNCTION
# ==============================
def train_model(model, train_loader, freeze_backbone=False):
    """
    Fine-tune model on the k-shot training loader.
    freeze_backbone=True  → only the classifier head is updated.
    freeze_backbone=False → all layers are updated (full fine-tuning).
    """
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
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    return model

# =========================================================
# FEW-SHOT LOOP
# k-shot samples are drawn ONLY from the adaptation pool.
# The fixed test set is NEVER touched during sampling.
# =========================================================
results = []

for shots in SHOT_LIST:

    print("\n" + "=" * 56)
    print(f"  SHOTS PER CLASS: {shots}  "
          f"(total train = {shots * len(class_names)})")
    print("=" * 56)

    # ── Sample k shots per class from the adaptation pool only ──
    # Reseed the per-shot RNG so each shot level is independently
    # reproducible but does not depend on previous iterations.
    shot_rng    = np.random.default_rng(SEED + shots)
    few_indices = []

    for c in range(len(class_names)):
        # Restrict to adaptation pool indices belonging to class c
        class_adapt_idx = np.array(
            [i for i in adapt_indices if targets[i] == c]
        )
        shot_rng.shuffle(class_adapt_idx)

        if shots > len(class_adapt_idx):
            print(f"  WARNING: Only {len(class_adapt_idx)} adapt images "
                  f"for class {c} — using all of them.")
            few_indices.extend(class_adapt_idx.tolist())
        else:
            few_indices.extend(class_adapt_idx[:shots].tolist())

    # Confirm zero overlap between train and test
    overlap = set(few_indices) & set(test_indices_fixed)
    assert len(overlap) == 0, \
        f"DATA LEAKAGE DETECTED: {len(overlap)} images appear in both train and test!"

    train_dataset = Subset(dataset, few_indices)
    train_loader  = DataLoader(train_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

    print(f"  Train size : {len(train_dataset)}  "
          f"(from adaptation pool only)")
    print(f"  Test size  : {len(fixed_test_dataset)}  "
          f"(fixed — identical across all shot levels)")

    # ── ZERO-SHOT ──
    print("\n---- Zero-Shot (no adaptation) ----")
    model = create_model(pretrained_tb11k=True)
    acc_z, f1_z, auc_z = evaluate(model, fixed_test_loader, class_names)
    print(f"Zero-shot : ACC={acc_z:.4f}  F1={f1_z:.4f}  AUC={auc_z:.4f}")

    # ── FROZEN BACKBONE ──
    print("\n---- Frozen Backbone Adaptation ----")
    model = create_model(pretrained_tb11k=True)
    model = train_model(model, train_loader, freeze_backbone=True)
    acc_fr, f1_fr, auc_fr = evaluate(model, fixed_test_loader, class_names)
    print(f"Frozen    : ACC={acc_fr:.4f}  F1={f1_fr:.4f}  AUC={auc_fr:.4f}")

    # ── FULL FINE-TUNE ──
    print("\n---- Full Fine-Tuning ----")
    model = create_model(pretrained_tb11k=True)
    model = train_model(model, train_loader, freeze_backbone=False)
    acc_fu, f1_fu, auc_fu = evaluate(model, fixed_test_loader, class_names)
    print(f"Full FT   : ACC={acc_fu:.4f}  F1={f1_fu:.4f}  AUC={auc_fu:.4f}")

    # ── SCRATCH ──
    print("\n---- Training from Scratch ----")
    model = create_model(pretrained_tb11k=False)
    model = train_model(model, train_loader, freeze_backbone=False)
    acc_sc, f1_sc, auc_sc = evaluate(model, fixed_test_loader, class_names)
    print(f"Scratch   : ACC={acc_sc:.4f}  F1={f1_sc:.4f}  AUC={auc_sc:.4f}")

    results.append({
        "shots":       shots,
        "frozen_acc":  acc_fr,  "frozen_auc":  auc_fr,
        "full_acc":    acc_fu,  "full_auc":    auc_fu,
        "scratch_acc": acc_sc,  "scratch_auc": auc_sc,
    })

# ==============================
# FINAL SUMMARY TABLE
# ==============================
print("\n\n" + "=" * 76)
print("FINAL SUMMARY  (all experiments use the SAME fixed test set)")
print("=" * 76)
print(f"{'Shots':>6} | {'Frozen ACC':>10} {'Frozen AUC':>10} | "
      f"{'Full ACC':>8} {'Full AUC':>8} | "
      f"{'Scratch ACC':>11} {'Scratch AUC':>11}")
print("-" * 76)
for r in results:
    print(f"{r['shots']:>6} | "
          f"{r['frozen_acc']:>10.4f} {r['frozen_auc']:>10.4f} | "
          f"{r['full_acc']:>8.4f} {r['full_auc']:>8.4f} | "
          f"{r['scratch_acc']:>11.4f} {r['scratch_auc']:>11.4f}")
print("=" * 76)
print(f"\nFixed test set: {len(fixed_test_dataset)} images")
print(f"Adaptation pool: {len(adapt_indices)} images")
print("No data leakage — verified by assertion in loop.")