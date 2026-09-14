import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================
# PATH TO YOUR DATASET
# ==========================
MENDELEY_PATH = r"Mendeley Dataset\Mendeley Data"   # <-- CHANGE if needed
MODEL_PATH = "best_tb11k_binary_densenet.pth"  # your saved model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ==========================
    # TRANSFORMS
    # ==========================
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ==========================
    # DATASET
    # ==========================
    test_dataset = datasets.ImageFolder(
        root=MENDELEY_PATH,
        transform=test_transform
    )

    print("\nDetected Classes:", test_dataset.classes)
    print("Class to index mapping:", test_dataset.class_to_idx)
    print("Total Test Images:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0   # IMPORTANT for Windows
    )

    # ==========================
    # LOAD MODEL
    # ==========================
# ==========================
# RECREATE TRAINING MODEL EXACTLY
# ==========================
    class DenseNetBinary(nn.Module):
        def __init__(self):
            super(DenseNetBinary, self).__init__()
            self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
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
    
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print("Model loaded successfully!")
    
        # ==========================
        # EVALUATION
        # ==========================
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # TB class probability

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ==========================
    # METRICS
    # ==========================
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=test_dataset.classes
    ))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    print(f"\nAUC Score: {roc_auc:.4f}")

    # ==========================
    # PLOT CONFUSION MATRIX
    # ==========================
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], test_dataset.classes, rotation=45)
    plt.yticks([0, 1], test_dataset.classes)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # ==========================
    # PLOT ROC CURVE
    # ==========================
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()