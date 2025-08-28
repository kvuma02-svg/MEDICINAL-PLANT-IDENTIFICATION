
import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models

from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, recall_score,
    confusion_matrix, matthews_corrcoef
)

# USER CONFIG
DATA_ROOT = r"Medicinal plant dataset"
TEST_IMAGE_PATH = r"Medicinal plant dataset\Aloevera\339.jpg"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training hyperparams
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

IMAGE_SIZE = 224   # ViT expects 224
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-4
NUM_WORKERS = 4
AMP = True

# Quick debug mode: run on small subset and fewer epochs (set False for full runs)
FAST_DEBUG = True
SUBSET_SIZE = 120   # only used if FAST_DEBUG True
if FAST_DEBUG:
    BATCH_SIZE = 8
    NUM_EPOCHS = 2
    NUM_WORKERS = 2

# Splits to evaluate: 0.3 => 70:30, 0.2 => 80:20
SPLITS = [0.3, 0.2]

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.12, 0.12, 0.08),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Model components
# -------------------------
class QuantumFeatureExtractor(nn.Module):
    """Quantum-inspired probabilistic feature mapping (classical simulation)."""
    def __init__(self, in_dim:int, out_dim:int=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        y = torch.tanh(self.fc(x))
        y = self.norm(y)
        probs = F.softmax(y, dim=-1)
        return y * probs

def get_resnet_backbone(proj_dim=512, pretrained=True, small=False):
    """Return backbone module that outputs features and projected embedding."""
    if small:
        res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    backbone = nn.Sequential(*list(res.children())[:-1])  # remove fc
    feat_dim = res.fc.in_features
    proj = nn.Linear(feat_dim, proj_dim)
    return backbone, proj, feat_dim

def get_vit_backbone(proj_dim=384, pretrained=True):
    """Return ViT backbone and projection (ViT expects 224x224)."""
    try:
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = vit.heads.head.in_features
        vit.heads = nn.Identity()
        proj = nn.Linear(feat_dim, proj_dim)
        return vit, proj, feat_dim
    except Exception:
        small = nn.Sequential(nn.Conv2d(3, 64, kernel_size=16, stride=16), nn.AdaptiveAvgPool2d(1))
        feat_dim = 64
        proj = nn.Linear(feat_dim, proj_dim)
        return small, proj, feat_dim

class MediFloraNet(nn.Module):
    """Fusion of Enhanced CNN (ResNet) + Improved ViT + Quantum features."""
    def __init__(self, num_classes:int, res_out=512, vit_out=384, q_out=128, use_small_backbone=False, pretrained=True):
        super().__init__()
        self.res_backbone, self.res_proj, self.res_feat_dim = get_resnet_backbone(res_out, pretrained=pretrained, small=use_small_backbone)
        self.vit_backbone, self.vit_proj, self.vit_feat_dim = get_vit_backbone(vit_out, pretrained=pretrained)
        self.qfe = QuantumFeatureExtractor(in_dim=vit_out, out_dim=q_out)
        fused_dim = res_out + vit_out + q_out
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ViT features
        vit_feat = self.vit_backbone(x)
        if isinstance(vit_feat, torch.Tensor) and vit_feat.ndim == 4:
            vit_feat = vit_feat.flatten(1)
        vit_emb = self.vit_proj(vit_feat) if hasattr(self, 'vit_proj') else vit_feat
        # ResNet features
        res_feat = self.res_backbone(x).flatten(1)
        res_emb = self.res_proj(res_feat)
        # Quantum features
        q_emb = self.qfe(vit_emb)
        fused = torch.cat([res_emb, vit_emb, q_emb], dim=1)
        return self.classifier(fused)

def compute_metrics_multiclass(y_true:List[int], y_pred:List[int]) -> Dict[str,float]:
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    sens = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    mcc = matthews_corrcoef(y_true, y_pred) * 100

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    specif_list, npv_list, fpr_list, fnr_list = [], [], [], []
    for i in range(n_classes):
        TP = cm[i,i]
        FN = cm[i,:].sum() - TP
        FP = cm[:,i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec = (TN / (TN + FP)) if (TN + FP) > 0 else 0.0
        npv = (TN / (TN + FN)) if (TN + FN) > 0 else 0.0
        fpr = (FP / (FP + TN)) if (FP + TN) > 0 else 0.0
        fnr = (FN / (FN + TP)) if (FN + TP) > 0 else 0.0
        specif_list.append(spec)
        npv_list.append(npv)
        fpr_list.append(fpr)
        fnr_list.append(fnr)
    specificity = np.mean(specif_list) * 100
    npv = np.mean(npv_list) * 100
    fpr = np.mean(fpr_list)
    fnr = np.mean(fnr_list)
    return {
        "Accuracy": acc,
        "Precision": prec,
        "F1-Score": f1,
        "Specificity": specificity,
        "Sensitivity": sens,
        "NPV": npv,
        "MCC": mcc,
        "FPR": fpr,
        "FNR": fnr
    }

scaler = torch.cuda.amp.GradScaler() if (AMP and DEVICE.type == 'cuda') else None

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def eval_model(model, loader, device):
    model.eval()
    ys_true, ys_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            ys_pred.extend(preds.tolist())
            ys_true.extend(labels.numpy().tolist())
    metrics = compute_metrics_multiclass(ys_true, ys_pred)
    return metrics, ys_true, ys_pred

# -------------------------
# Data loaders (subset support)
# -------------------------
def prepare_loaders(data_root:str, val_split:float=0.2, batch_size:int=8, subset_size:Optional[int]=None):
    dataset = datasets.ImageFolder(root=data_root, transform=train_transform)
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    if subset_size and subset_size < n:
        indices = indices[:subset_size]
    split = int(np.floor(val_split * len(indices)))
    train_idx, val_idx = indices[split:], indices[:split]
    train_ds = Subset(dataset, train_idx)
    val_base = datasets.ImageFolder(root=data_root, transform=val_transform)
    val_ds = Subset(val_base, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=='cuda'))
    return train_loader, val_loader, dataset.classes

# -------------------------
# Single image prediction (prints true & predicted)
# -------------------------
def predict_single_image(model, img_path:str, class_list:List[str], device=DEVICE):
    model.eval()
    true_label = Path(img_path).parent.name
    img = Image.open(img_path).convert('RGB')
    img_t = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
        pred = out.argmax(dim=1).item()
    pred_label = class_list[pred]
    print(f"True label     : {true_label}")
    print(f"Predicted label: {pred_label}")
    return true_label, pred_label

# -------------------------
# Main pipeline
# -------------------------
def run_pipeline():
    results = []
    for split in SPLITS:
        print(f"\n=== Running split test_size={split} (train:{100-int(split*100)} val:{int(split*100)}) ===")
        train_loader, val_loader, class_list = prepare_loaders(DATA_ROOT, val_split=split,
                                                               batch_size=BATCH_SIZE,
                                                               subset_size=(SUBSET_SIZE if FAST_DEBUG else None))
        num_classes = len(class_list)
        print(f"Found {len(train_loader.dataset) + len(val_loader.dataset)} samples (train+val), classes={num_classes}")

        model = MediFloraNet(num_classes=num_classes,
                             res_out=512, vit_out=384, q_out=128,
                             use_small_backbone=FAST_DEBUG, pretrained=not FAST_DEBUG).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

        best_acc = 0.0
        best_path = os.path.join(CHECKPOINT_DIR, f"best_mediflora_split_{int((1-split)*100)}_{int(split*100)}.pth")
        for epoch in range(1, NUM_EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            metrics, _, _ = eval_model(model, val_loader, DEVICE)
            t1 = time.time()
            print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | val_acc {metrics['Accuracy']:.2f}% | time {(t1-t0):.1f}s")
            if metrics['Accuracy'] > best_acc:
                best_acc = metrics['Accuracy']
                torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}, best_path)
                print(f"Saved best checkpoint -> {best_path} (val acc {best_acc:.4f})")

        # load best and final evaluate
        if os.path.exists(best_path):
            ck = torch.load(best_path, map_location=DEVICE)
            model.load_state_dict(ck['model_state'])
        final_metrics, y_true, y_pred = eval_model(model, val_loader, DEVICE)
        rec = {"Model": "Proposed MediFlora-Net", "Split": f"{int((1-split)*100)}:{int(split*100)}"}
        rec.update(final_metrics)
        results.append(rec)

        # save predictions for this split
        dfp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        dfp.to_csv(os.path.join(CHECKPOINT_DIR, f"predictions_split_{int((1-split)*100)}_{int(split*100)}.csv"), index=False)

    # Save aggregate results
    df = pd.DataFrame(results)
    csv_path = os.path.join(CHECKPOINT_DIR, "mediflora_performance.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}\n")
    print(df.to_string(index=False))

    # single image prediction if exists
    if os.path.exists(TEST_IMAGE_PATH):
        print("\nSingle image test result:")
        predict_single_image(model, TEST_IMAGE_PATH, class_list, DEVICE)
    else:
        print("\nTest image not found, skipping single-image prediction.")

run_pipeline()
