#!/usr/bin/env python3
"""
Evaluate a trained DANN model:
  • Loads best_dann.pth and PCA+Scaler (joblib)
  • Computes accuracy, ROC-AUC, PR-AUC
  • Optionally plots t-SNE and curves

Usage:
  python evaluate.py --data_path <.mat/.npz> \
                     --model_path best_dann.pth \
                     --pca_scaler pca_scaler.joblib \
                     --out_dir results/ --device cpu
"""

import os, argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.manifold import TSNE
from joblib import load
import matplotlib.pyplot as plt

# --------- Import model definition from train.py --------- #
from train import DANN, set_seed, flatten_nonzero_voxels

def load_data(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        data = loadmat(path)
        X_train = data["X_train"]
        y_train = data["y_train"].ravel()
        X_test  = data["X_test"]
        y_test  = data["y_test"].ravel()
    elif ext == ".npz":
        data = np.load(path, allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train"].ravel()
        X_test  = data["X_test"]
        y_test  = data["y_test"].ravel()
    else:
        raise ValueError("Unsupported data format. Use .mat or .npz")
    return X_train, y_train, X_test, y_test

def evaluate(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")

    # Load raw data and mask/flatten
    X_train, y_train, X_test, y_test = load_data(args.data_path)
    _, X_test = flatten_nonzero_voxels(X_train, X_test)

    # Load scaler + PCA, then transform test
    pack = load(args.pca_scaler)  # {"scaler":..., "pca":...}
    scaler, pca = pack["scaler"], pack["pca"]
    X_test_s = scaler.transform(X_test)
    X_test_p = pca.transform(X_test_s)

    in_dim = X_test_p.shape[1]

    # DataLoader
    test_ds = TensorDataset(torch.from_numpy(X_test_p).float(), torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = DANN(in_dim=in_dim, hidden=args.hidden_dim, n_classes=2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Evaluate
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits, _ = model(xb, alpha=0.0)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(yb.numpy())

    logits = np.vstack(all_logits)
    y_true = np.concatenate(all_labels)
    y_prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
    y_pred = logits.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

    # Curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    # Save plots
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC = {roc_auc:.4f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "roc_curve.png")); plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve (AP = {ap:.4f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "pr_curve.png")); plt.close()

    # t-SNE of feature space
    # (Compute features explicitly to visualize)
    feats = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            h = model.feature(xb)
            feats.append(h.cpu().numpy())
    X_feat = np.vstack(feats)
    X_2d = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=args.seed).fit_transform(X_feat)

    plt.figure()
    plt.scatter(X_2d[y_true == 0, 0], X_2d[y_true == 0, 1], s=10, label="Class 0")
    plt.scatter(X_2d[y_true == 1, 0], X_2d[y_true == 1, 1], s=10, label="Class 1")
    plt.legend()
    plt.xlabel("TSNE-1"); plt.ylabel("TSNE-2"); plt.title("t-SNE of DANN Features (Test)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "tsne_features.png")); plt.close()

    # Write summary
    summary = {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] metrics.json and plots to {args.out_dir}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help=".mat or .npz with X_train,y_train,X_test,y_test")
    ap.add_argument("--model_path", type=str, default="best_dann.pth")
    ap.add_argument("--pca_scaler", type=str, default="pca_scaler.joblib")
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=48)
    ap.add_argument("--device", type=str, choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--out_dir", type=str, default="results")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

