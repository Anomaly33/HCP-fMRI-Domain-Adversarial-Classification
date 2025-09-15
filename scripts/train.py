#!/usr/bin/env python3
"""
Train a Domain Adversarial Neural Network (DANN) for HCP fMRI
face-vs-shape classification with RobustScaler + PCA.

Inputs can be either:
  • a .mat file containing X_train, y_train, X_test, y_test
  • a .npz file containing X_train, y_train, X_test, y_test
The target (X_test) is used unlabeled for domain alignment.

Saves:
  • best_dann.pth             (model weights)
  • pca_scaler.joblib         (RobustScaler + PCA)
  • metrics_train.json        (optional if you add logging)
"""

import os, json, random, argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# ---------------------- Utils ---------------------- #
def set_seed(seed: int = 48):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(path: str):
    """
    Returns (X_train, y_train, X_test, y_test).
    Accepts .mat with variables or .npz with same keys.
    """
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

def flatten_nonzero_voxels(X_train, X_test):
    """
    Follow the notebook’s approach:
    Use a mask from the first sample to keep only non-zero voxels,
    then flatten per sample.
    """
    mask = (X_train[0] != 0)
    X_train_flat = np.vstack([X_train[i][mask] for i in range(X_train.shape[0])])
    X_test_flat  = np.vstack([X_test[i][mask]  for i in range(X_test.shape[0])])
    return X_train_flat, X_test_flat

# ---------------------- Model ---------------------- #
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(True),
            nn.Linear(1024, hidden), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class LabelPredictor(nn.Module):
    def __init__(self, hidden: int = 512, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(True),
            nn.Linear(256, n_classes)
        )
    def forward(self, h): return self.net(h)

class DomainDiscriminator(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(True),
            nn.Linear(256, 2)
        )
    def forward(self, h, alpha: float):
        h_rev = GradReverse.apply(h, alpha)
        return self.net(h_rev)

class DANN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, n_classes: int = 2):
        super().__init__()
        self.feature = FeatureExtractor(in_dim, hidden)
        self.label   = LabelPredictor(hidden, n_classes)
        self.domain  = DomainDiscriminator(hidden)
    def forward(self, x, alpha: float = 0.0):
        h = self.feature(x)
        y = self.label(h)
        d = self.domain(h, alpha)
        return y, d

# ---------------------- Training ---------------------- #
def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")

    # Load & preprocess
    X_train, y_train, X_test, y_test = load_data(args.data_path)

    # Flatten masked voxels (based on notebook)
    X_train, X_test = flatten_nonzero_voxels(X_train, X_test)

    # Robust scaling + PCA
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    pca = PCA(n_components=args.pca_components, svd_solver="randomized", whiten=True, random_state=args.seed)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p  = pca.transform(X_test_s)
    in_dim = X_train_p.shape[1]
    print(f"[Info] PCA features: {in_dim}")

    # Train/val split on source
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_p, y_train, test_size=0.1, stratify=y_train, random_state=args.seed
    )

    # Datasets & loaders
    train_src = TensorDataset(torch.from_numpy(X_tr).float(),  torch.from_numpy(y_tr).long())
    val_src   = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    # target is unlabeled (use only X)
    train_tgt = TensorDataset(torch.from_numpy(X_test_p).float(), torch.zeros(len(X_test_p)).long())

    src_loader = DataLoader(train_src, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    tgt_loader = DataLoader(train_tgt, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_src,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model / losses / optim
    dann = DANN(in_dim=in_dim, hidden=args.hidden_dim, n_classes=2).to(device)
    cls_loss = nn.CrossEntropyLoss()
    dom_loss = nn.CrossEntropyLoss()
    optim_dann = optim.Adam(dann.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    total_steps = args.epochs * min(len(src_loader), len(tgt_loader))
    step = 0

    for epoch in range(1, args.epochs + 1):
        dann.train()
        for (xs, ys), (xt, _) in zip(src_loader, tgt_loader):
            p = step / max(total_steps, 1)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # GRL schedule
            step += 1

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            y_pred_s, d_pred_s = dann(xs, alpha)
            _,        d_pred_t = dann(xt, alpha)

            L_cls = cls_loss(y_pred_s, ys)
            L_dom = dom_loss(d_pred_s, torch.zeros(len(d_pred_s), dtype=torch.long, device=device)) \
                  + dom_loss(d_pred_t, torch.ones(len(d_pred_t), dtype=torch.long, device=device))
            loss = L_cls + L_dom

            optim_dann.zero_grad()
            loss.backward()
            optim_dann.step()

        # Validation (source domain)
        dann.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out, _ = dann(xb, alpha=0.0)
                preds.append(out.argmax(1).cpu().numpy())
                labels.append(yb.numpy())
        val_acc = accuracy_score(np.concatenate(labels), np.concatenate(preds))
        print(f"[Epoch {epoch:02d}] Source Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(dann.state_dict(), args.save_model)
            print(f"  ↳ Saved best model to {args.save_model}")

    # Persist scaler + PCA
    dump({"scaler": scaler, "pca": pca}, args.save_pca_scaler)
    print(f"[Done] Saved scaler+PCA to {args.save_pca_scaler}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help=".mat or .npz with X_train,y_train,X_test,y_test")
    ap.add_argument("--save_model", type=str, default="best_dann.pth")
    ap.add_argument("--save_pca_scaler", type=str, default="pca_scaler.joblib")
    ap.add_argument("--pca_components", type=int, default=500)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=48)
    ap.add_argument("--device", type=str, choices=["cpu","cuda"], default="cpu")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)

