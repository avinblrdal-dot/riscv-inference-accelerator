#!/usr/bin/env python3
"""Train a model from a frozen config.

REPRODUCIBILITY IS THE POINT
----------------------------
Every source of randomness is seeded from the config, and the seed plus the
git SHA plus the config hash are written into the checkpoint. Given a
checkpoint you can always answer "what code, what config, what seed produced
this?" -- which is the question a judge or a reviewer will actually ask.

The config must be FROZEN before training. Training against an unfrozen config
would produce a model nobody can reproduce, so this script refuses to start.

Usage:
    python3 train/train.py --config train/config/workload_a.yaml --synthetic-data
    python3 train/train.py --config train/config/workload_a.yaml \
        --data data/cache/workload_a_fault_classifier.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from freeze import require_frozen, config_hash, git_sha  # noqa: E402


def seed_everything(seed: int) -> None:
    """Seed every RNG we might touch.

    Missing one of these is the usual reason "identical" runs diverge. numpy
    and Python's random are used by the data pipeline; torch by the model
    init and the shuffler; cudnn's autotuner by GPU kernel selection.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic kernels cost some speed but make runs comparable.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, bool]:
    z = np.load(path, allow_pickle=False)
    synthetic = bool(z["synthetic"]) if "synthetic" in z.files else False
    return z["X"], z["y"], synthetic


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", help="preprocessed .npz from train/data.py")
    ap.add_argument("--synthetic-data", action="store_true",
                    help="generate synthetic data on the fly")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the config (for smoke tests only -- a "
                         "reported result must use the frozen value)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    cfg = require_frozen(args.config)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("ERROR: PyTorch is required to train.\n"
              "  pip install -r train/requirements.txt\n\n"
              "  Everything downstream of training -- quantization, export,\n"
              "  bit-exactness, RTL simulation -- works WITHOUT torch. Use\n"
              "  'python3 train/quantize.py --synthetic' to exercise it.",
              file=sys.stderr)
        return 2

    from models import build_model, count_macs

    seed = cfg["training"]["seed"]
    seed_everything(seed)

    out_dir = args.out_dir or os.path.join(HERE, "runs", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f" Training {cfg['name']}")
    print("=" * 70)
    print(f"  seed         : {seed}")
    print(f"  config hash  : {config_hash(cfg)[:16]}...")
    print(f"  git          : {git_sha()}")
    print(f"  MACs/inference: {count_macs(cfg):,}")
    print()

    # ---- data ----
    if args.synthetic_data:
        from data import synthesize_dataset
        print("  DATA: SYNTHETIC -- accuracy from this run is NOT a result.")
        X, y = synthesize_dataset(cfg, n_per_class=128, seed=seed)
        synthetic = True
    elif args.data:
        X, y, synthetic = load_dataset(args.data)
        if synthetic:
            print("  DATA: the file is tagged SYNTHETIC -- accuracy is not a result.")
    else:
        ap.error("give --data or --synthetic-data")

    print(f"  dataset: X={X.shape} y={y.shape}")

    # Stratified split so every class appears in validation. A plain random
    # split can leave a rare class entirely out of validation, which makes the
    # reported accuracy meaningless for that class.
    rng = np.random.default_rng(seed)
    val_frac = cfg["training"]["val_split"]
    tr_idx, va_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        va_idx += idx[:n_val].tolist()
        tr_idx += idx[n_val:].tolist()
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)

    # Normalise using TRAIN statistics only. Using the whole dataset's mean
    # and std would leak validation information into training and inflate the
    # reported accuracy -- a subtle and very common mistake.
    mu = float(X[tr_idx].mean())
    sd = float(X[tr_idx].std()) or 1.0
    Xn = ((X - mu) / sd).astype(np.float32)

    dev = torch.device(args.device)
    Xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(y)

    train_ds = TensorDataset(Xt[tr_idx], yt[tr_idx])
    val_ds = TensorDataset(Xt[va_idx], yt[va_idx])
    bs = cfg["training"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # ---- model ----
    model = build_model(cfg).to(dev)
    is_autoencoder = cfg["model"]["architecture"] == "fc_autoencoder"
    criterion = nn.L1Loss() if is_autoencoder else nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),
                           lr=cfg["training"]["learning_rate"],
                           weight_decay=cfg["training"].get("weight_decay", 0.0))

    epochs = args.epochs or cfg["training"]["epochs"]
    patience = cfg["training"].get("early_stopping_patience", 10)
    best_metric, best_epoch, since_best = -1.0, -1, 0
    history = []
    t_start = time.time()

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, xb.flatten(1)) if is_autoencoder \
                else criterion(out, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss) * xb.size(0)
        tr_loss /= max(len(train_ds), 1)

        model.eval()
        correct, total, va_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                if is_autoencoder:
                    va_loss += float(criterion(out, xb.flatten(1))) * xb.size(0)
                else:
                    va_loss += float(criterion(out, yb)) * xb.size(0)
                    correct += int((out.argmax(1) == yb).sum())
                total += xb.size(0)
        va_loss /= max(total, 1)
        acc = correct / total if (total and not is_autoencoder) else 0.0

        # For the autoencoder, lower loss is better; for the classifier,
        # higher accuracy is.
        metric = -va_loss if is_autoencoder else acc
        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_acc": acc})

        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  train={tr_loss:.4f}  val={va_loss:.4f}"
                  + ("" if is_autoencoder else f"  acc={acc*100:.2f}%"))

        if metric > best_metric:
            best_metric, best_epoch, since_best = metric, ep, 0
            torch.save({
                "state_dict": model.state_dict(),
                "config_name": cfg["name"],
                "config_sha256": config_hash(cfg),
                "git_sha": git_sha(),
                "seed": seed,
                "epoch": ep,
                "accuracy": acc,
                "val_loss": va_loss,
                "norm_mean": mu,
                "norm_std": sd,
                "synthetic_data": synthetic,
            }, os.path.join(out_dir, "best.pt"))
        else:
            since_best += 1
            if since_best >= patience:
                print(f"  early stop at epoch {ep} "
                      f"(no improvement for {patience} epochs)")
                break

    elapsed = time.time() - t_start
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump({"history": history, "best_epoch": best_epoch,
                   "seed": seed, "synthetic_data": synthetic,
                   "elapsed_s": elapsed}, fh, indent=2)

    print()
    print(f"  best epoch {best_epoch}, "
          + ("val_loss %.4f" % -best_metric if is_autoencoder
             else "val acc %.2f%%" % (best_metric * 100)))
    print(f"  wrote {out_dir}/best.pt  ({elapsed:.1f}s)")
    if synthetic:
        print()
        print("  REMINDER: trained on SYNTHETIC data. This number is not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
