import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

# =========================
# SETTINGS
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"D:\Research Work\TB Classification\Mendeley Dataset\Mendeley Data"
TB11K_MODEL_PATH = r"D:\Research Work\TB Classification\best_tb11k_binary_densenet.pth"

BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 5
LR = 1e-4
SHOT = 50
SEEDS = [42, 52, 62]

print("Device:", DEVICE)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

dataset = ImageFolder(DATA_ROOT, transform=transform)
targets = np.array(dataset.targets)

# =========================
# PATIENT GROUPING (CRITICAL)
# =========================
def get_patient_id(path):
    filename = os.path.basename(path)
    return filename.split('_')[0]   # ⚠️ MODIFY if needed

groups = defaultdict(list)
for idx, (path, _) in enumerate(dataset.samples):
    pid = get_patient_id(path)
    groups[pid].append(idx)

print("Total patients (approx):", len(groups))

# =========================
# SPLIT FUNCTION
# =========================
def split_by_patient(groups, seed):
    random.seed(seed)
    patients = list(groups.keys())
    random.shuffle(patients)

    n = len(patients)
    train_p = patients[:int(0.6*n)]
    val_p   = patients[int(0.6*n):int(0.8*n)]
    test_p  = patients[int(0.8*n):]

    def expand(p_list):
        idx = []
        for p in p_list:
            idx.extend(groups[p])
        return idx

    return expand(train_p), expand(val_p), expand(test_p)

# =========================
# MODEL
# =========================
def create_model():
    model = models.densenet121(weights=None)

    # load TBX11K pretrained backbone
    state_dict = torch.load(TB11K_MODEL_PATH, map_location=DEVICE)
    new_state = {k.replace("model.",""):v for k,v in state_dict.items() if "classifier" not in k}
    model.load_state_dict(new_state, strict=False)

    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model.to(DEVICE)

# =========================
# TRAIN FUNCTION (WITH EARLY STOPPING)
# =========================
def train(model, train_loader, val_loader, class_weights):

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_auc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        val_auc = evaluate(model, val_loader, silent=True)

        print(f"Epoch {epoch+1} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model)
    return model

# =========================
# EVALUATE
# =========================
def evaluate(model, loader, silent=False):
    model.eval()
    preds, probs, labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            p = torch.softmax(out,1)[:,1]

            preds.extend(torch.argmax(out,1).cpu().numpy())
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    if not silent:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)

        print(f"ACC={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
        print(f"Sensitivity={sens:.4f} Specificity={spec:.4f}")

    return auc

# =========================
# MAIN EXPERIMENT
# =========================
all_results = []

for seed in SEEDS:
    print("\n====================")
    print("SEED:", seed)

    train_idx, val_idx, test_idx = split_by_patient(groups, seed)

    # FEW SHOT sampling from TRAIN ONLY
    few_idx = []
    for c in [0,1]:
        c_idx = [i for i in train_idx if targets[i]==c]
        random.shuffle(c_idx)
        few_idx.extend(c_idx[:SHOT])

    # loaders
    train_loader = DataLoader(Subset(dataset, few_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE)

    # class imbalance handling
    class_counts = np.bincount(targets[few_idx])
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE)

    model = create_model()
    model = train(model, train_loader, val_loader, weights)

    print("TEST RESULT:")
    auc = evaluate(model, test_loader)
    all_results.append(auc)

# =========================
# FINAL STATS
# =========================
mean_auc = np.mean(all_results)
std_auc  = np.std(all_results)

print("\n====================")
print(f"FINAL AUC: {mean_auc:.4f} ± {std_auc:.4f}")