import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np, random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MENDELEY_ROOT = r"D:\Research Work\TB Classification\Mendeley Dataset\Mendeley Data"

# Destroy anatomy: downsample hard, then upsample back to input size
tf_blur = transforms.Compose([
    transforms.Resize((8, 8)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

dataset = ImageFolder(root=MENDELEY_ROOT, transform=tf_blur)
targets = np.array(dataset.targets)
class_names = dataset.classes
print("class_to_idx:", dataset.class_to_idx)

rng = np.random.default_rng(42)
adapt_idx, test_idx = [], []
for c in range(len(class_names)):
    ci = np.where(targets == c)[0].copy(); rng.shuffle(ci)
    n = int(len(ci) * 0.20)
    adapt_idx.extend(ci[:n].tolist()); test_idx.extend(ci[n:].tolist())

test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False)
tt = targets[test_idx]
print(f"test n={len(test_idx)}  majority={np.bincount(tt).max()/len(tt)*100:.2f}%")

def create_model():
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m.to(DEVICE)

def evaluate(model):
    model.eval(); P,Q,L = [],[],[]
    with torch.no_grad():
        for x,y in test_loader:
            o = model(x.to(DEVICE))
            Q.extend(torch.softmax(o,1)[:,1].cpu().numpy())
            P.extend(o.argmax(1).cpu().numpy()); L.extend(y.numpy())
    P,Q,L = np.array(P),np.array(Q),np.array(L)
    tn,fp,fn,tp = confusion_matrix(L,P).ravel()
    return dict(acc=accuracy_score(L,P), auc=roc_auc_score(L,Q),
                spec=tn/(tn+fp), sens=tp/(tp+fn))

def train(model, loader, epochs=10, lr=1e-4):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x,y in loader:
            opt.zero_grad(); crit(model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
    return model

adapt_idx = np.array(adapt_idx)
adapt_targets = targets[adapt_idx]

for k in [10, 25, 50]:
    for d in range(3):
        seed = 1000*d + 42
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        drng = np.random.default_rng(seed)
        few = []
        for c in range(len(class_names)):
            ci = adapt_idx[adapt_targets==c].copy(); drng.shuffle(ci)
            few.extend(ci[:min(k,len(ci))].tolist())
        loader = DataLoader(Subset(dataset, few), batch_size=16, shuffle=True)
        m = train(create_model(), loader)
        r = evaluate(m)
        print(f"BLUR k={k:3d} draw={d}  acc={r['acc']*100:.2f}  auc={r['auc']:.4f}  spec={r['spec']:.3f}")