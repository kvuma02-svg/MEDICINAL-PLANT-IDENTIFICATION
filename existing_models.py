
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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, recall_score,
    confusion_matrix, matthews_corrcoef
)
from tqdm import tqdm
from models import _CNN, _ViT, CNN_ViT_Fusion, EnhancedCNN, ImprovedViT, MediFloraNet , ViT_ReT_, get_swin_, BrainNPT_, iEEG_HCT_

DATA_ROOT = r"Medicinal plant dataset"
TEST_IMAGE_PATH = r"Medicinal plant dataset\Aloevera\339.jpg"
CHECKPOINT_DIR = "./checkpoints_all_models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-4
NUM_WORKERS = 4
AMP = True and (DEVICE.type == 'cuda')

FAST_DEBUG = True
SUBSET_SIZE = 240
if FAST_DEBUG:
    BATCH_SIZE = 8
    NUM_EPOCHS = 1
    NUM_WORKERS = 0  # <- Change this to 0 for Windows stability


SPLITS = [0.3,  0.2]
# SPLITS = [ 0.2]

MODEL_NAMES = [
    "CNN",
    "ViT",
    "CNN-ViT",
    "enhanced CNN",
    "improved ViT",
    "Proposed MediFlora-Net",
    "ViT-ReT",
    "Swin-CFNet",
    "BrainNPT",
    "iEEG-HCT"
]

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

def prepare_loaders(data_root: str, val_split: float = 0.2, batch_size: int = 8,
                    subset_size: Optional[int] = None):
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

def compute_metrics_multiclass(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    sens = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    mcc = matthews_corrcoef(y_true, y_pred) * 100

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    specif_list, npv_list, fpr_list, fnr_list = [], [], [], []
    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
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


scaler = torch.cuda.amp.GradScaler() if AMP else None

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        # handle ImageFolder returning (img, label)
        if len(batch) == 2:
            imgs, labels = batch
        else:
            imgs, _, labels = batch
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
        for batch in loader:
            if len(batch) == 2:
                imgs, labels = batch
            else:
                imgs, _, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            ys_pred.extend(preds.tolist())
            ys_true.extend(labels.cpu().numpy().tolist())
    metrics = compute_metrics_multiclass(ys_true, ys_pred)
    return metrics, ys_true, ys_pred

def run_model(name: str, model_obj: nn.Module, train_loader, val_loader, class_list: List[str], split_label: str):
    print(f"\n--- Training model: {name} ---")
    model = model_obj.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    best_acc = 0.0
    best_ckpt = os.path.join(CHECKPOINT_DIR, f"{name.replace(' ','_')}_best_{split_label}.pth")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        metrics, _, _ = eval_model(model, val_loader, DEVICE)
        t1 = time.time()
        print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | val_acc {metrics['Accuracy']:.2f}% | time {(t1-t0):.1f}s")
        if metrics['Accuracy'] > best_acc:
            best_acc = metrics['Accuracy']
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}, best_ckpt)
    # load best and compute final metrics
    if os.path.exists(best_ckpt):
        ck = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(ck['model_state'])
    final_metrics, y_true, y_pred = eval_model(model, val_loader, DEVICE)
    # free GPU memory
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    return final_metrics, y_true, y_pred

def build_model_(name: str, num_classes: int, small_backbone: bool):
    name = name.lower()
    if name == "cnn":
        return _CNN(num_classes)
    if name == "vit":
        return _ViT(num_classes)
    if name == "cnn-vit":
        return CNN_ViT_Fusion(num_classes, use_small=small_backbone)
    if name == "enhanced cnn":
        return EnhancedCNN(num_classes, small=small_backbone)
    if name == "improved vit":
        return ImprovedViT(num_classes)
    if name == "proposed mediflora-net":
        return MediFloraNet(num_classes, small=small_backbone)
    if name == "vit-ret":
        return ViT_ReT_(num_classes)
    if name == "swin-cfnet":
        return get_swin_(num_classes)
    if name == "brainnpt":
        return BrainNPT_(num_classes)
    if name == "ieeg-hct":
        return iEEG_HCT_(num_classes)
    # default fallback
    return _CNN(num_classes)

# -------------------------
# Main: run all models across splits and save CSVs
# -------------------------
def run_all():
    aggregate_results = []
    for split in SPLITS:
        split_label = f"{int((1-split)*100)}_{int(split*100)}"
        print(f"\n=== Running split test_size={split} (train:{100-int(split*100)} val:{int(split*100)}) ===")
        train_loader, val_loader, class_list = prepare_loaders(DATA_ROOT, val_split=split,
                                                               batch_size=BATCH_SIZE,
                                                               subset_size=(SUBSET_SIZE if FAST_DEBUG else None))
        num_classes = len(class_list)
        print(f"Found {len(train_loader.dataset) + len(val_loader.dataset)} samples (train+val), {num_classes} classes")

        results_for_split = []
        for model_name in MODEL_NAMES:
            print(f"\nRunning model: {model_name}")
            # build model
            model_obj = build_model_(model_name, num_classes, small_backbone=FAST_DEBUG)
            metrics, y_true, y_pred = run_model(model_name, model_obj, train_loader, val_loader, class_list, split_label)
            rec = {"Model": model_name, "Split": f"{int((1-split)*100)}:{int(split*100)}"}
            rec.update(metrics)
            results_for_split.append(rec)
            # save predictions for this model & split
            pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            fn = os.path.join(CHECKPOINT_DIR, f"preds_{model_name.replace(' ','_')}_{split_label}.csv")
            pred_df.to_csv(fn, index=False)
            print(f"Saved preds -> {fn}")

        # save per-split CSV
        dfsplit = pd.DataFrame(results_for_split)
        out_csv = os.path.join(CHECKPOINT_DIR, f"results_split_{split_label}.csv")
        dfsplit.to_csv(out_csv, index=False)
        print(f"Saved split results -> {out_csv}")
        aggregate_results.extend(results_for_split)

    # save aggregate CSV
    df_all = pd.DataFrame(aggregate_results)
    master_csv = os.path.join(CHECKPOINT_DIR, "mediflora_all_models_results.csv")
    df_all.to_csv(master_csv, index=False)
    print(f"\nSaved master CSV -> {master_csv}")
    print(df_all.to_string(index=False))

    # single-image test using last-trained model (Proposed MediFlora-Net) if exists
    if os.path.exists(TEST_IMAGE_PATH):
        # attempt to load last best Proposed model for last split
        ckpt_glob = os.path.join(CHECKPOINT_DIR, f"Proposed_MediFlora-Net_best_*")
        # fallback: just instantiate model & do prediction (untrained) to show API
        model_inst = MediFloraNet(num_classes=num_classes, small=FAST_DEBUG).to(DEVICE)
        model_inst.eval()
        img = Image.open(TEST_IMAGE_PATH).convert('RGB')
        img_t = val_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model_inst(img_t)
            pred = out.argmax(dim=1).item()
        print(f"\nSingle-image test : Predicted class index: {pred} -> label: {class_list[pred]}")


import torch.multiprocessing as mp
mp.freeze_support()  # Required for Windows multiprocessing
run_all()
