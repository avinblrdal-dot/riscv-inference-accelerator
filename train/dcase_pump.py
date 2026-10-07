#!/usr/bin/env python3
"""Real data for workload B: the MIMII pump recordings (DCASE 2020 Task 2 split).

WHY THIS DATASET
----------------
The application is failure detection on off-grid water handpumps
(docs/DECISIONS.md D023). No public handpump dataset is known. The closest
public stand-in is the MIMII pump subset -- recordings of real INDUSTRIAL
pumps, normal and malfunctioning. Any number produced here must say so: it
measures how well the deployed model detects pump anomalies in general, not
handpump failures specifically.

We use the DCASE 2020 Challenge Task 2 development split of MIMII pump
(Zenodo record 3678171, dev_data_pump.zip, 1.03 GB, CC BY-NC-SA 4.0) rather
than the raw MIMII archive (7.66 GB) because:
  * it is the same recordings, pre-split into train (normal only) and test
    (normal + anomalous) by the challenge organisers, so the split is not
    ours to tune; and
  * it has a published baseline to compare against, scored with the same
    metrics computed here (AUC and pAUC at max FPR 0.1, averaged over the
    four machine IDs).
Citation: Koizumi et al., "Description and Discussion on DCASE2020 Challenge
Task2", DCASE 2020; Purohit et al., "MIMII Dataset", DCASE 2019.

FEATURES
--------
Exactly the frozen config's input block (train/config/workload_b.yaml):
16 kHz audio, n_fft = 1024, hop = 512, first n_bins = 128 FFT magnitude bins
(0 - 2 kHz). The deployed model's input is ONE frame (input_dim = 128); the
config's n_frames_stacked = 4 applies only to the host-only full model.
Magnitudes are log-compressed, log(1 + |X|). The config says "fft_magnitude"
without fixing linear vs log; log is the standard choice for audio anomaly
detection and is recorded as a decision in docs/DECISIONS.md D025.

A clip's anomaly score is the mean, over all its frames, of the per-frame
reconstruction error -- the same per-frame number the firmware computes.

Usage (from the repo root, inside .venv):
    python3 train/dcase_pump.py prepare              # wavs -> data/cache/*.npz
    python3 train/train.py --config train/config/workload_b.yaml \
        --data data/cache/dcase_pump_train.npz --out-dir train/runs/workload_b_real
    python3 train/quantize.py --config train/config/workload_b.yaml \
        --checkpoint train/runs/workload_b_real/best.pt \
        --calib-data data/cache/dcase_pump_calib.npz \
        --out train/runs/workload_b_real/quantized.npz
    python3 train/dcase_pump.py evaluate             # float AND int8 AUC
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import quant_ref as qr  # noqa: E402

RAW_ZIP = os.path.join(ROOT, "data", "raw", "mimii", "dev_data_pump.zip")
RAW_DIR = os.path.join(ROOT, "data", "raw", "mimii", "pump")
CACHE = os.path.join(ROOT, "data", "cache")
RUN = os.path.join(HERE, "runs", "workload_b_real")
SEED = 20260828            # workload_b.yaml training.seed

SR, N_FFT, HOP, N_BINS = 16000, 1024, 512, 128
TRAIN_FRAME_STRIDE = 4     # keep every 4th frame of a training clip
THRESHOLD_CLIP_FRAC = 0.1  # normal train clips held out for the threshold


def clip_features(path: str) -> np.ndarray:
    """(n_frames, 128) log-magnitude FFT frames for one wav."""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: {sr} Hz, expected {SR}")
    x = wav[:, 0]
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(N_FFT).astype(np.float32)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1))[:, :N_BINS]
    return np.log1p(mag).astype(np.float32)


def parse_name(path: str) -> tuple[str, str]:
    m = re.match(r"(normal|anomaly)_(id_\d+)_\d+\.wav", os.path.basename(path))
    if not m:
        raise ValueError(f"unexpected file name {path}")
    return m.group(1), m.group(2)


def prepare() -> int:
    if not os.path.isdir(RAW_DIR):
        if not os.path.exists(RAW_ZIP):
            print(f"ERROR: {RAW_ZIP} not found", file=sys.stderr)
            return 1
        print(f"Unzipping {RAW_ZIP} ...")
        with zipfile.ZipFile(RAW_ZIP) as zf:
            zf.extractall(os.path.dirname(RAW_ZIP))

    train_files = sorted(glob.glob(os.path.join(RAW_DIR, "train", "*.wav")))
    test_files = sorted(glob.glob(os.path.join(RAW_DIR, "test", "*.wav")))
    print(f"  train clips: {len(train_files)}   test clips: {len(test_files)}")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train_files))
    n_hold = int(len(train_files) * THRESHOLD_CLIP_FRAC)
    hold = set(order[:n_hold].tolist())

    tr_frames, hold_clips, hold_ids = [], [], []
    for i, f in enumerate(train_files):
        lab, mid = parse_name(f)
        assert lab == "normal", f"train split must be normal only: {f}"
        feats = clip_features(f)
        if i in hold:
            hold_clips.append(feats)
            hold_ids.append(mid)
        else:
            tr_frames.append(feats[::TRAIN_FRAME_STRIDE])
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(train_files)} train clips")
    Xtr = np.concatenate(tr_frames)

    te_clips, te_lab, te_ids = [], [], []
    for f in test_files:
        lab, mid = parse_name(f)
        te_clips.append(clip_features(f))
        te_lab.append(1 if lab == "anomaly" else 0)
        te_ids.append(mid)

    os.makedirs(CACHE, exist_ok=True)
    # Training frames: normal only (train_on_normal_only: true), y = 0.
    np.savez_compressed(os.path.join(CACHE, "dcase_pump_train.npz"),
                        X=Xtr, y=np.zeros(len(Xtr), dtype=np.int64),
                        synthetic=np.array(False))
    # Calibration: a seeded sample of training frames (raw, un-normalised).
    cal = Xtr[rng.permutation(len(Xtr))[:1024]]
    np.savez_compressed(os.path.join(CACHE, "dcase_pump_calib.npz"), X=cal)
    np.savez_compressed(os.path.join(CACHE, "dcase_pump_eval.npz"),
                        hold=np.stack(hold_clips), hold_ids=np.array(hold_ids),
                        test=np.stack(te_clips), test_label=np.array(te_lab),
                        test_ids=np.array(te_ids))
    print(f"  training frames: {Xtr.shape}  held-out normal clips: {len(hold_clips)}"
          f"  test clips: {len(te_clips)} ({sum(te_lab)} anomalous)")
    return 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def float_scores(clips: np.ndarray, model, mu: float, sd: float) -> np.ndarray:
    """Clip score = mean over frames of per-frame mean |recon - input|."""
    import torch
    n_clip, n_fr, d = clips.shape
    x = ((clips.reshape(-1, d) - mu) / sd).astype(np.float32)
    with torch.no_grad():
        r = model(torch.from_numpy(x)).numpy()
    return np.abs(r - x).mean(axis=1).reshape(n_clip, n_fr).mean(axis=1)


def int8_scores(clips: np.ndarray, q: dict, mu: float, sd: float,
                in_scale: float) -> np.ndarray:
    """The SAME arithmetic as the chip, batched over frames.

    Input quantized with the calibrated input scale; every layer is
    quant_ref.linear_int + requantize (+ relu), the reference that sw/ and rtl/
    are proven bit-exact against; per-frame score is the firmware's own
    sum|out - in| / 128 with integer division (sw/src/main.c, measure()).
    """
    n_clip, n_fr, d = clips.shape
    x = (clips.reshape(-1, d) - mu) / sd
    xq = qr.quantize_tensor(x, in_scale, 0, 8).astype(np.int64)
    cur = xq
    for li, L in enumerate(q["layers"]):
        acc = cur @ L["weight"].astype(np.int64).T + L["bias"].astype(np.int64)
        cur = qr.requantize(acc, L["multiplier"], L["shift"], 0, 8).astype(np.int64)
        if li < len(q["layers"]) - 1:          # relu after every fc but the last
            cur = np.maximum(cur, 0)
    frame = np.abs(cur - xq).sum(axis=1) // d
    return frame.reshape(n_clip, n_fr).mean(axis=1)


def auc_report(scores, labels, ids) -> dict:
    from sklearn.metrics import roc_auc_score
    out = {}
    for mid in sorted(set(ids)):
        m = np.array(ids) == mid
        out[mid] = {"auc": float(roc_auc_score(labels[m], scores[m])),
                    "pauc": float(roc_auc_score(labels[m], scores[m], max_fpr=0.1))}
    out["mean"] = {"auc": float(np.mean([v["auc"] for k, v in out.items()])),
                   "pauc": float(np.mean([v["pauc"] for k, v in out.items()
                                          if k != "mean"]))}
    return out


def detection_at_threshold(test_s, test_lab, hold_s, pct: float) -> dict:
    thr = float(np.percentile(hold_s, pct))
    pred = test_s > thr
    lab = np.asarray(test_lab).astype(bool)
    return {"threshold": thr,
            "detected_anomalies": float(pred[lab].mean()),
            "false_alarms_on_normal": float(pred[~lab].mean())}


def evaluate() -> int:
    import torch
    from config import load_config
    from models import build_model

    cfg = load_config(os.path.join(HERE, "config", "workload_b.yaml"))
    ckpt = torch.load(os.path.join(RUN, "best.pt"), map_location="cpu",
                      weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mu, sd = float(ckpt["norm_mean"]), float(ckpt["norm_std"])

    z = np.load(os.path.join(CACHE, "dcase_pump_eval.npz"))
    test, lab, ids = z["test"], z["test_label"], z["test_ids"]
    hold = z["hold"]

    qz = np.load(os.path.join(RUN, "quantized.npz"))
    q = {"layers": [{"weight": qz[f"L{i}_weight"], "bias": qz[f"L{i}_bias"],
                     "multiplier": int(qz[f"L{i}_multiplier"]),
                     "shift": int(qz[f"L{i}_shift"])}
                    for i in range(int(qz["n_layers"]))]}
    in_scale = float(qz["act_scale_0"])

    pct = float(cfg["detection"]["threshold_percentile"])
    res = {"dataset": "DCASE2020 Task2 dev, MIMII pump (industrial pumps, NOT handpumps)",
           "n_test_clips": int(len(test)), "n_test_anomalous": int(lab.sum()),
           "n_threshold_clips": int(len(hold))}
    for name, fn in (("float", lambda c: float_scores(c, model, mu, sd)),
                     ("int8_chip_arithmetic",
                      lambda c: int8_scores(c, q, mu, sd, in_scale))):
        ts, hs = fn(test), fn(hold)
        res[name] = {"auc": auc_report(ts, lab, ids),
                     "detection_at_p%g" % pct: detection_at_threshold(ts, lab, hs, pct)}

    os.makedirs(RUN, exist_ok=True)
    with open(os.path.join(RUN, "eval.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    print(json.dumps(res, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "evaluate"])
    a = ap.parse_args()
    return prepare() if a.step == "prepare" else evaluate()


if __name__ == "__main__":
    sys.exit(main())
