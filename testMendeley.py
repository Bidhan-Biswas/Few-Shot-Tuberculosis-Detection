import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import densenet121
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ==============================
# PATHS
# ==============================
MODEL_PATH = "tb_densenet_weighted_sampler.pth"
MENDELEY_PATH = "Mendeley Dataset/Mendeley Data"

# ==============================
# TRANSFORM (MUST MATCH TRAINING)
# ==============================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==============================
# LOAD DATASET
# ==============================
test_dataset = datasets.ImageFolder(
    root=MENDELEY_PATH,
    transform=test_transform
)

print("Class mapping:", test_dataset.class_to_idx)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,   # keep small (memory safe)
    shuffle=False
)

# ==============================
# LOAD TRAINED MODEL
# ==============================
model = densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully")

# ==============================
# EVALUATION
# ==============================
all_binary_preds = []
all_labels = []
all_tb_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1)

        # Convert 3-class → Binary
        binary_preds = (preds != 0).int()

        # TB probability = latent + active
        tb_probs = probs[:,1] + probs[:,2]

        all_binary_preds.extend(binary_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_tb_probs.extend(tb_probs.cpu().numpy())

# ==============================
# METRICS
# ==============================
print("\n===== EXTERNAL VALIDATION RESULTS =====")

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_binary_preds,
    target_names=["Normal", "TB"]
))

cm = confusion_matrix(all_labels, all_binary_preds)
print("\nConfusion Matrix:")
print(cm)

# Sensitivity & Specificity
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\nSensitivity (Recall TB):", sensitivity)
print("Specificity (Recall Normal):", specificity)

auc = roc_auc_score(all_labels, all_tb_probs)
print("ROC-AUC:", auc)

accuracy = np.mean(np.array(all_binary_preds) == np.array(all_labels))
print("Accuracy:", accuracy)