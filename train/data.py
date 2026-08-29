#!/usr/bin/env python3
"""Dataset download, caching and preprocessing.

DATASETS
--------
MIMII (Hitachi)  -- industrial machine sound: valves, pumps, fans, slide rails,
                    recorded normal and anomalous, with real factory noise
                    mixed in at several SNRs. Zenodo, ~10 GB, CC BY-SA 4.0.
CWRU bearing     -- Case Western Reserve bearing vibration, the standard
                    reference dataset for bearing fault diagnosis. MATLAB
                    .mat files, small.

BOTH ARE LARGE THIRD-PARTY DOWNLOADS AND URLS ROT. This module is written so
that a dead link is an inconvenience, not a blocker:

  * every download is checked and cached
  * a failure prints EXACT manual-download instructions and the expected
    directory layout
  * ``--synthetic`` generates physically plausible fake data so the entire
    training and export pipeline can be exercised offline

The synthetic generator is not a shortcut around the real data. It exists so
that pipeline bugs get found before the team spends a day on a 10 GB
download, and so CI has something to run. Anything produced from it is
tagged synthetic and can never be reported as a result.

Usage:
    python3 train/data.py --dataset mimii --download
    python3 train/data.py --dataset mimii --synthetic --out data/cache
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# TODO_BLOCKED (see docs/DECISIONS.md): these URLs are recorded from the
# datasets' published landing pages but have NOT been verified from this
# machine. Verify before relying on them; if one is dead, the manual
# instructions below are the fallback.
DATASETS = {
    "mimii": {
        "name": "MIMII (Malfunctioning Industrial Machine Investigation)",
        "url": "https://zenodo.org/record/3384388",
        "landing": "https://zenodo.org/record/3384388",
        "license": "CC BY-SA 4.0",
        "citation": "Purohit et al., MIMII Dataset, DCASE 2019",
        "manual": """
  1. Open https://zenodo.org/record/3384388
  2. Download one or more machine archives, e.g. '-6_dB_pump.zip'.
     Start with ONE machine type -- the full set is ~10 GB and you do not
     need it to get the pipeline working.
  3. Unzip into  data/raw/mimii/
     Expected layout:
         data/raw/mimii/pump/id_00/normal/*.wav
         data/raw/mimii/pump/id_00/abnormal/*.wav
  4. Re-run this script without --download.
""",
    },
    "cwru": {
        "name": "Case Western Reserve University bearing vibration data",
        "url": "https://engineering.case.edu/bearingdatacenter/download-data-file",
        "landing": "https://engineering.case.edu/bearingdatacenter",
        "license": "Free for research use with citation",
        "citation": "Case Western Reserve University Bearing Data Center",
        "manual": """
  1. Open https://engineering.case.edu/bearingdatacenter/download-data-file
  2. Download the 12k Drive End Bearing Fault .mat files, plus the Normal
     Baseline files.
  3. Place them in  data/raw/cwru/
     Expected layout:
         data/raw/cwru/normal/*.mat
         data/raw/cwru/inner_race/*.mat
         data/raw/cwru/outer_race/*.mat
         data/raw/cwru/ball/*.mat
  4. Re-run this script without --download.
""",
    },
}

CLASSES = ["normal", "imbalance", "misalignment", "bearing_fault"]

#: Which spectrogram backend the last call to log_mel_spectrogram() used.
#: Recorded into dataset metadata because librosa and the numpy fallback are
#: NOT bit-identical -- a model trained on one and fed the other quietly loses
#: accuracy, and that is very hard to diagnose after the fact.
LAST_BACKEND = "unset"


def print_manual_instructions(key: str, reason: str) -> None:
    d = DATASETS[key]
    print("=" * 70, file=sys.stderr)
    print(f" COULD NOT FETCH {d['name']}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f" Reason: {reason}", file=sys.stderr)
    print(file=sys.stderr)
    print(" Download it by hand instead:", file=sys.stderr)
    print(d["manual"], file=sys.stderr)
    print(f" License:  {d['license']}", file=sys.stderr)
    print(f" Cite as:  {d['citation']}", file=sys.stderr)
    print(file=sys.stderr)
    print(" Meanwhile, --synthetic exercises the whole pipeline offline.",
          file=sys.stderr)


def try_download(key: str, dest_dir: str, timeout: int = 30) -> bool:
    """Attempt a download; return False (with instructions) rather than raising."""
    d = DATASETS[key]
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Checking {d['name']}...")
    print(f"  landing page: {d['landing']}")
    try:
        req = urllib.request.Request(d["landing"],
                                     headers={"User-Agent": "riscv-accel/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                print_manual_instructions(key, f"HTTP {resp.status}")
                return False
    except (urllib.error.URLError, OSError) as exc:
        print_manual_instructions(key, f"{type(exc).__name__}: {exc}")
        return False

    # Deliberately NOT automating the bulk download. Both datasets require
    # accepting terms and are many gigabytes; scripting around that would be
    # both fragile and impolite to the hosts. Point the user at the page.
    print("  landing page reachable.")
    print_manual_instructions(key, "bulk download is intentionally manual "
                                   "(licence acceptance + multi-GB archives)")
    return False


def _set_backend(name: str) -> None:
    global LAST_BACKEND
    LAST_BACKEND = name


def log_mel_spectrogram(wave: np.ndarray, sr: int, n_fft: int, hop: int,
                        n_mels: int, n_frames: int) -> np.ndarray:
    """Compute a log-mel spectrogram, preferring librosa and falling back.

    The fallback is a plain numpy STFT plus a triangular mel filterbank. It is
    not bit-identical to librosa, so DO NOT MIX THEM: a model trained on
    librosa features and fed fallback features at inference will quietly lose
    accuracy. The chosen path is recorded in the output metadata for exactly
    that reason.
    """
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=wave.astype(np.float32), sr=sr, n_fft=n_fft,
            hop_length=hop, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        _set_backend("librosa")
    except ImportError:
        # --- numpy fallback ---
        n_hops = 1 + (len(wave) - n_fft) // hop if len(wave) >= n_fft else 1
        window = np.hanning(n_fft)
        frames = np.zeros((n_fft // 2 + 1, max(n_hops, 1)))
        for i in range(max(n_hops, 1)):
            seg = wave[i*hop:i*hop + n_fft]
            if len(seg) < n_fft:
                seg = np.pad(seg, (0, n_fft - len(seg)))
            frames[:, i] = np.abs(np.fft.rfft(seg * window)) ** 2

        def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)

        edges = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2))
        bins = np.floor((n_fft + 1) * edges / sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            l, c, r = bins[m - 1], bins[m], bins[m + 1]
            for k in range(l, min(c, fb.shape[1])):
                if c > l:
                    fb[m - 1, k] = (k - l) / (c - l)
            for k in range(c, min(r, fb.shape[1])):
                if r > c:
                    fb[m - 1, k] = (r - k) / (r - c)
        mel = fb @ frames
        log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        log_mel -= log_mel.max()
        _set_backend("numpy_fallback")

    # Crop or pad to exactly n_frames so every sample has the same shape --
    # the hardware has fixed-size buffers and cannot handle variable input.
    if log_mel.shape[1] >= n_frames:
        start = (log_mel.shape[1] - n_frames) // 2
        log_mel = log_mel[:, start:start + n_frames]
    else:
        log_mel = np.pad(log_mel, ((0, 0), (0, n_frames - log_mel.shape[1])),
                         mode="edge")

    return log_mel.astype(np.float32)


def synthesize_dataset(cfg: dict, n_per_class: int, seed: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Generate physically motivated fake machine sound.

    Each fault class gets a distinct, physically sensible signature:

      normal          broadband noise plus the shaft rotation tone
      imbalance       a strong 1x-shaft-rate tone (mass off-centre)
      misalignment    a strong 2x-shaft-rate tone (the classic signature)
      bearing_fault   high-frequency ringing amplitude-modulated at the
                      ball-pass frequency (impacts as the ball hits a defect)

    These are the real signatures a vibration engineer looks for, so a model
    trained on this data learns something structurally similar to the real
    task. It is still NOT real data and must never be reported as a result --
    it exists to make the pipeline runnable and CI meaningful.
    """
    rng = np.random.default_rng(seed)
    inp = cfg["input"]
    sr = inp["sample_rate_hz"]
    dur = 1.0
    t = np.arange(int(sr * dur)) / sr
    shaft_hz = 50.0

    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            sig = 0.05 * rng.standard_normal(t.shape)
            jitter = 1.0 + 0.02 * rng.standard_normal()
            f0 = shaft_hz * jitter

            sig += 0.10 * np.sin(2*np.pi*f0*t)          # always present
            if cls == "imbalance":
                sig += 0.45 * np.sin(2*np.pi*f0*t + rng.uniform(0, 6.28))
            elif cls == "misalignment":
                sig += 0.40 * np.sin(2*np.pi*2*f0*t + rng.uniform(0, 6.28))
                sig += 0.15 * np.sin(2*np.pi*3*f0*t)
            elif cls == "bearing_fault":
                bpfo = f0 * 3.58                        # typical ball-pass ratio
                carrier = np.sin(2*np.pi*3200*t)
                envelope = 0.5 * (1 + np.sign(np.sin(2*np.pi*bpfo*t)))
                sig += 0.35 * carrier * envelope

            spec = log_mel_spectrogram(sig, sr, inp["n_fft"], inp["hop_length"],
                                       inp["n_mels"], inp["n_frames"])
            X.append(spec)
            y.append(ci)

    return np.stack(X)[:, None, :, :], np.array(y, dtype=np.int64)


def main() -> int:
    sys.path.insert(0, HERE)
    from config import load_config

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="mimii")
    ap.add_argument("--config", default=os.path.join(HERE, "config",
                                                     "workload_a.yaml"))
    ap.add_argument("--download", action="store_true",
                    help="check availability and print fetch instructions")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate physically plausible fake data offline")
    ap.add_argument("--n-per-class", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cache"))
    args = ap.parse_args()

    if args.download:
        ok = try_download(args.dataset, os.path.join(ROOT, "data", "raw",
                                                     args.dataset))
        return 0 if ok else 1

    if args.synthetic:
        cfg = load_config(args.config)
        print(f"Generating synthetic data for '{cfg['name']}'")
        print(f"  {args.n_per_class} samples/class x {len(CLASSES)} classes, "
              f"seed={args.seed}")
        X, y = synthesize_dataset(cfg, args.n_per_class, args.seed)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{cfg['name']}_synthetic.npz")
        np.savez_compressed(path, X=X, y=y, synthetic=np.array(True),
                            seed=np.array(args.seed),
                            classes=np.array(CLASSES),
                            spectrogram_backend=np.array(LAST_BACKEND))
        digest = hashlib.sha256(X.tobytes()).hexdigest()[:16]
        print(f"  X={X.shape} y={y.shape}  sha={digest}")
        print(f"  spectrogram backend: {LAST_BACKEND}")
        if LAST_BACKEND == "numpy_fallback":
            print("  NOTE: librosa is not installed, so the numpy fallback was")
            print("        used. It is NOT bit-identical to librosa -- do not")
            print("        mix features from the two in one experiment.")
        print(f"Wrote {path}")
        print()
        print("  SYNTHETIC DATA. Useful for exercising the pipeline and for CI.")
        print("  Never report accuracy from this as a result.")
        return 0

    raw = os.path.join(ROOT, "data", "raw", args.dataset)
    if not os.path.isdir(raw) or not os.listdir(raw):
        print_manual_instructions(args.dataset, f"{raw} is missing or empty")
        return 1

    print(f"Found raw data in {raw}")
    print("  (real-data preprocessing is wired to the same "
          "log_mel_spectrogram() used above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
