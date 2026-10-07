#!/usr/bin/env python3
"""Does a pump sensor get better if it learns its OWN pump after installation?

THE QUESTION (exploratory -- see docs/DECISIONS.md D027)
-------------------------------------------------------
A model trained in the lab meets a pump it has never heard. Every pump sounds
a little different (age, mounting, wear), and the first real-data run showed
it: one of the four MIMII pumps (id_04) scored far below the others (D025).
A handpump sensor has no internet, no engineer and no labelled failures -- but
an anomaly detector only needs NORMAL sound to learn from, and the first days
after installation are (presumably) normal. So: let the sensor adapt itself
on a short recording of its own pump, and see whether detection improves.

THE PROTOCOL: leave-one-pump-out, so "a pump it has never heard" is literal
  for each held-out pump P in {id_00, id_02, id_04, id_06}:
    1. train the deployed workload-B model (frozen architecture and
       hyperparameters) on the OTHER three pumps' normal clips only
    2. score P's test clips with no adaptation              -> "unadapted"
    3. adapt on the first N normal clips of P's train split, then score
       P's test clips again (P's test clips are never used to adapt)
  adaptation methods, cheapest first:
    renorm     re-estimate only the input mean/std on P's clips (2 numbers)
    last       fine-tune only the final layer (32x128 + 128 = 4,224 params)
    full       fine-tune every layer (8,904 params)
  N = 10, 30, 100 clips (10 s each: 1.7, 5 and 17 minutes of pumping)

Every score is reported twice: float, and int8 using the chip's arithmetic
(quant_ref, recalibrated on the same adaptation clips -- also something a
device could do). This simulates the LEARNING in float on the host; doing the
learning itself in integer arithmetic on the chip is the next step, not this.

Usage (repo root, inside .venv; needs `train/dcase_pump.py prepare` first):
    python3 train/adapt_experiment.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import quant_ref as qr                     # noqa: E402
from config import load_config             # noqa: E402
from dcase_pump import (RAW_DIR, clip_features, parse_name,  # noqa: E402
                        TRAIN_FRAME_STRIDE)
from quantize import quantize_model         # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache", "dcase_pump_byid.npz")
OUT_DIR = os.path.join(HERE, "runs", "adapt_experiment")
RESULTS_MD = os.path.join(ROOT, "docs", "results", "ON_DEVICE_ADAPTATION.md")
PUMPS = ["id_00", "id_02", "id_04", "id_06"]
N_ADAPT = [10, 30, 100]


# ---------------------------------------------------------------------------
# Data, kept per pump this time
# ---------------------------------------------------------------------------
def load_by_pump() -> dict:
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}
    print("Extracting features per pump (one-off, a few minutes)...")
    out = {}
    for split in ("train", "test"):
        files = sorted(glob.glob(os.path.join(RAW_DIR, split, "*.wav")))
        feats, ids, labs = [], [], []
        for f in files:
            lab, mid = parse_name(f)
            feats.append(clip_features(f))
            ids.append(mid)
            labs.append(1 if lab == "anomaly" else 0)
        out[f"{split}_X"] = np.stack(feats)        # (clips, frames, 128)
        out[f"{split}_id"] = np.array(ids)
        out[f"{split}_y"] = np.array(labs)
    np.savez_compressed(CACHE, **out)
    return out


# ---------------------------------------------------------------------------
# Training: same architecture, optimiser, lr, batch, patience and epoch cap as
# train/train.py and the frozen config -- only the data differs. A manual
# loop over in-memory tensors replaces DataLoader, which dominated train.py's
# runtime for this tiny model (45 min for one run).
# ---------------------------------------------------------------------------
def train_model(cfg, X: np.ndarray, seed: int, init_state=None, epochs=None,
                params="all", mu=None, sd=None):
    import torch
    import torch.nn as nn
    from models import build_model

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tc = cfg["training"]
    if mu is None:
        mu, sd = float(X.mean()), float(X.std()) or 1.0
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32))

    model = build_model(cfg)
    if init_state is not None:
        model.load_state_dict(init_state)
    if params == "last":
        for p in model.parameters():
            p.requires_grad_(False)
        last = [m for m in model if isinstance(m, nn.Linear)][-1]
        for p in last.parameters():
            p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=tc["learning_rate"])
    loss_fn = nn.L1Loss()

    idx = rng.permutation(len(Xn))
    n_val = max(1, int(len(idx) * tc["val_split"]))
    va, tr = idx[:n_val], idx[n_val:]
    bs = tc["batch_size"]
    max_ep = epochs or tc["epochs"]
    best, best_state, since = float("inf"), None, 0
    for ep in range(max_ep):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        for i in range(0, len(perm), bs):
            xb = Xn[perm[i:i + bs]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(Xn[va]), Xn[va]))
        if v < best - 1e-6:
            best, since = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            since += 1
            if since >= tc["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, mu, sd


# ---------------------------------------------------------------------------
# Scoring, float and int8 (chip arithmetic)
# ---------------------------------------------------------------------------
def float_scores(model, clips, mu, sd):
    import torch
    n, f, d = clips.shape
    x = ((clips.reshape(-1, d) - mu) / sd).astype(np.float32)
    with torch.no_grad():
        r = model(torch.from_numpy(x)).numpy()
    return np.abs(r - x).mean(1).reshape(n, f).mean(1)


def int8_quantize(model, cfg, calib_frames, mu, sd) -> tuple[dict, float]:
    """Calibrate on `calib_frames` exactly as quantize.py --calib-data does."""
    import torch
    x = (calib_frames[:256].astype(np.float64) - mu) / sd
    scales = [qr.choose_scale(x, 8)]
    mods = list(model)
    with torch.no_grad():
        cur = torch.from_numpy(x.astype(np.float32))
        for i, m in enumerate(mods):
            cur = m(cur)
            if isinstance(m, torch.nn.Linear):
                relu_next = i + 1 < len(mods) and isinstance(mods[i + 1], torch.nn.ReLU)
                scales.append(qr.choose_scale((torch.relu(cur) if relu_next else cur).numpy(), 8))
    scales[-1] = scales[0]          # output on the input's scale (see quantize.py)
    layers = [{"type": "fc", "weight": m.weight.detach().numpy(),
               "bias": m.bias.detach().numpy()}
              for m in mods if isinstance(m, torch.nn.Linear)]
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        q = quantize_model({"layers": layers, "synthetic": False}, cfg, scales)
    return q, scales[0]


def int8_scores(q, in_scale, clips, mu, sd):
    n, f, d = clips.shape
    xq = qr.quantize_tensor((clips.reshape(-1, d) - mu) / sd, in_scale, 0, 8).astype(np.int64)
    cur = xq
    for li, L in enumerate(q["layers"]):
        acc = cur @ L["weight"].astype(np.int64).T + L["bias"].astype(np.int64)
        cur = qr.requantize(acc, L["multiplier"], L["shift"], 0, 8).astype(np.int64)
        if li < len(q["layers"]) - 1:
            cur = np.maximum(cur, 0)
    return (np.abs(cur - xq).sum(1) // d).reshape(n, f).mean(1)


def aucs(scores, labels):
    from sklearn.metrics import roc_auc_score
    return (float(roc_auc_score(labels, scores)),
            float(roc_auc_score(labels, scores, max_fpr=0.1)))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--adapt-epochs", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(os.path.join(HERE, "config", "workload_b.yaml"))
    base_seed = int(cfg["training"]["seed"])
    d = load_by_pump()
    trX, trid = d["train_X"], d["train_id"]
    teX, teid, tey = d["test_X"], d["test_id"], d["test_y"]
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    t0 = time.time()
    for s in range(args.seeds):
        seed = base_seed + s
        for P in PUMPS:
            others = trX[trid != P][:, ::TRAIN_FRAME_STRIDE].reshape(-1, trX.shape[2])
            base, mu, sd = train_model(cfg, others, seed)
            base_state = {k: t.clone() for k, t in base.state_dict().items()}
            tX, ty = teX[teid == P], tey[teid == P]
            own = trX[trid == P]
            own = own[np.random.default_rng(seed).permutation(len(own))]

            def record(method, n, model, m_mu, m_sd, calib):
                fa = aucs(float_scores(model, tX, m_mu, m_sd), ty)
                q, isc = int8_quantize(model, cfg, calib, m_mu, m_sd)
                ia = aucs(int8_scores(q, isc, tX, m_mu, m_sd), ty)
                rows.append({"seed": seed, "pump": P, "method": method, "n_clips": n,
                             "float_auc": fa[0], "float_pauc": fa[1],
                             "int8_auc": ia[0], "int8_pauc": ia[1]})
                print(f"  seed {seed} {P} {method:9s} n={n:>3}  "
                      f"float AUC {fa[0]*100:5.1f}  int8 AUC {ia[0]*100:5.1f}   "
                      f"[{time.time()-t0:5.0f}s]", flush=True)

            calib_base = others[np.random.default_rng(seed).permutation(len(others))[:256]]
            record("unadapted", 0, base, mu, sd, calib_base)
            for n in N_ADAPT:
                A = own[:n].reshape(-1, own.shape[2])      # every frame of n clips
                a_mu, a_sd = float(A.mean()), float(A.std()) or 1.0
                record("renorm", n, base, a_mu, a_sd, A)
                for method in ("last", "full"):
                    m, _, _ = train_model(cfg, A, seed, init_state=base_state,
                                          epochs=args.adapt_epochs,
                                          params="last" if method == "last" else "all",
                                          mu=mu, sd=sd)
                    record(method, n, m, mu, sd, A)

    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    write_md(rows, args)
    print(f"\nwrote {RESULTS_MD}")
    return 0


def write_md(rows, args):
    import statistics as st
    L = ["# On-device adaptation experiment (exploratory, D027)", "",
         "Leave-one-pump-out on MIMII pump (DCASE 2020 Task 2 split). Each row: a model "
         "trained on the OTHER three pumps, installed on a pump it never heard, then "
         "adapted on N of that pump's normal clips (10 s each). Scores are on that pump's "
         "test clips, which are never used for adaptation. Mean ± standard deviation "
         f"over {args.seeds} seed(s). AUC in %, int8 = the chip's arithmetic.", "",
         "**These are industrial pumps, not handpumps** (D023). The learning is simulated "
         "in float on a laptop; only the scoring uses chip arithmetic.", ""]
    methods = [("unadapted", 0)] + [(m, n) for n in N_ADAPT for m in ("renorm", "last", "full")]
    L.append("| Method | Clips | " + " | ".join(PUMPS) + " | Mean |")
    L.append("|---|---|" + "---|" * (len(PUMPS) + 1))
    for m, n in methods:
        cells = [m, str(n)]
        means = []
        for P in PUMPS:
            v = [r["int8_auc"] * 100 for r in rows if r["pump"] == P and r["method"] == m and r["n_clips"] == n]
            means.append(v)
            cells.append(f"{st.mean(v):.1f}" + (f" ± {st.stdev(v):.1f}" if len(v) > 1 else ""))
        per_seed = [st.mean(x) for x in zip(*means)]
        cells.append(f"**{st.mean(per_seed):.1f}**" + (f" ± {st.stdev(per_seed):.1f}" if len(per_seed) > 1 else ""))
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "Method key: `unadapted` = straight out of the box. `renorm` = re-measure "
          "only the input's average level and spread on this pump (2 numbers). `last` = "
          "retrain only the final layer (4,224 weights). `full` = retrain every layer "
          "(8,904 weights). Full per-row data, including float and pAUC: "
          "`train/runs/adapt_experiment/results.json` (regenerable).", ""]
    os.makedirs(os.path.dirname(RESULTS_MD), exist_ok=True)
    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
