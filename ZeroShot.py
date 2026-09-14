import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
import numpy as np, random, json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MENDELEY_ROOT    = r"D:\Research Work\TB Classification\Mendeley Dataset\Mendeley Data"
TB11K_MODEL_PATH = r"D:\Research Work\TB Classification\best_tb11k_binary_densenet.pth"

BATCH_SIZE, LR, EPOCHS = 16, 1e-4, 10
SHOT_LIST   = [10, 25, 50, 75, 100]
DRAWS       = 10
ADAPT_RATIO = 0.20

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

dataset = ImageFolder(root=MENDELEY_ROOT, transform=transform)
class_names, targets = dataset.classes, np.array(dataset.targets)
print("class_to_idx:", dataset.class_to_idx)

# ---- fixed split (same RNG as your ZeroShot run, so the test set matches) ----
rng = np.random.default_rng(42)
adapt_indices, test_indices = [], []
for c in range(len(class_names)):
    ci = np.where(targets == c)[0].copy(); rng.shuffle(ci)
    n = int(len(ci) * ADAPT_RATIO)
    adapt_indices.extend(ci[:n].tolist()); test_indices.extend(ci[n:].tolist())

test_loader = DataLoader(Subset(dataset, test_indices), batch_size=64, shuffle=False)
tt = targets[test_indices]
MAJORITY = np.bincount(tt).max() / len(tt)
print(f"test n={len(test_indices)}  majority={MAJORITY*100:.2f}%")

# ---- model factory: source head architecture preserved ----------------------
def make_head():
    return nn.Sequential(nn.Linear(1024,512), nn.ReLU(inplace=True),
                         nn.Dropout(0.5), nn.Linear(512,2))

def create_model(init="tb11k"):
    m = models.densenet121(weights=None)
    m.classifier = make_head()
    if init == "tb11k":
        sd = torch.load(TB11K_MODEL_PATH, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        sd = {k.replace("model.","",1): v for k,v in sd.items()}
        miss, unexp = m.load_state_dict(sd, strict=False)
        assert not miss and not unexp, f"load failed: {miss[:3]} {unexp[:3]}"
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
    return dict(acc=accuracy_score(L,P), f1=f1_score(L,P), auc=roc_auc_score(L,Q),
                sens=tp/(tp+fn), spec=tn/(tn+fp), pred_tb=float(P.mean()),
                cm=[int(tn),int(fp),int(fn),int(tp)])

def train(model, loader, freeze=False):
    if freeze:
        for p in model.features.parameters(): p.requires_grad = False
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=LR)
    crit = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for x,y in loader:
            opt.zero_grad(); crit(model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
    return model

# ---- zero-shot: once, outside the loop -------------------------------------
zs = evaluate(create_model("tb11k"))
print("\n=== ZERO-SHOT ===", json.dumps({k:v for k,v in zs.items() if k!="cm"}, indent=2))

# ---- sweep ------------------------------------------------------------------
results = {"zero_shot": zs, "majority": float(MAJORITY), "runs": {}}
adapt_targets = targets[adapt_indices]
adapt_indices = np.array(adapt_indices)

for k in SHOT_LIST:
    for regime in ["frozen","full","scratch"]:
        runs = []
        for d in range(DRAWS):
            seed = 1000*d + 42
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            drng = np.random.default_rng(seed)
            few = []
            for c in range(len(class_names)):
                ci = adapt_indices[adapt_targets==c].copy(); drng.shuffle(ci)
                few.extend(ci[:min(k,len(ci))].tolist())
            assert not (set(few) & set(test_indices)), "LEAKAGE"
            loader = DataLoader(Subset(dataset,few), batch_size=BATCH_SIZE, shuffle=True)
            m = create_model("scratch" if regime=="scratch" else "tb11k")
            runs.append(evaluate(train(m, loader, freeze=(regime=="frozen"))))
        results["runs"][f"{k}_{regime}"] = runs
        a=[r["acc"] for r in runs]; s=[r["spec"] for r in runs]; u=[r["auc"] for r in runs]
        print(f"k={k:3d} {regime:8s} acc {np.mean(a)*100:5.2f}±{np.std(a)*100:4.2f}  "
              f"spec {np.mean(s):.3f}  auc {np.mean(u):.3f}")

json.dump(results, open("results_v2.json","w"), indent=2)
print("\nsaved results_v2.json")