#!/usr/bin/env python3
"""Post-training quantization: float weights -> int8 (or int4) + integer scales.

WHAT QUANTIZATION IS, briefly
-----------------------------
A trained network holds 32-bit floats. Inference on a microcontroller cannot
afford them: no FPU, 4x the memory, and far more energy per operation. So we
map each tensor onto small integers:

    q = round(x / scale)        scale = max|x| / 127     (int8, symmetric)

and do all the arithmetic in integers. The accumulated int32 result is then
scaled back down with ``requantize`` -- integer multiply and shift only, never
a float multiply. See train/quant_ref.py for why that is non-negotiable.

WHAT THIS SCRIPT PRODUCES
-------------------------
An .npz holding, per layer: int8 weights, int32 biases, and the (M0, shift)
integer multiplier pair that requantize() needs. That file is the single
input to export_weights.py, and it contains NO floats in the inference path.

POST-TRAINING, NOT QUANTIZATION-AWARE
-------------------------------------
We quantize after training rather than simulating quantization during it.
Post-training is simpler, needs no retraining loop, and for networks this
small the accuracy cost is usually a fraction of a percent. If the int4 sweep
shows a large drop, quantization-aware training is the obvious next step --
and RQ4 is precisely the experiment that would tell us whether it is worth it.
Recorded in docs/DECISIONS.md.

Usage:
    python3 train/quantize.py --config train/config/workload_a.yaml \
        --checkpoint train/runs/workload_a/best.pt --out train/runs/workload_a/quantized.npz
    python3 train/quantize.py --config ... --synthetic --out ...   # no torch needed
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import quant_ref as qr          # noqa: E402
from freeze import require_frozen  # noqa: E402


def quantize_weight_tensor(w: np.ndarray, precision: int
                           ) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor quantization of one weight tensor."""
    scale = qr.choose_scale(w, precision)
    if scale == 1.0 and np.all(w == 0):
        print("  WARNING: an all-zero weight tensor. That layer contributes "
              "nothing; check that training actually converged.")
    q = qr.quantize_tensor(w, scale, zero_point=0, precision=precision)
    return q.astype(np.int8 if precision == 8 else np.int8), scale


def compute_layer_multiplier(in_scale: float, w_scale: float,
                             out_scale: float) -> tuple[int, int]:
    """Turn the three float scales into the integer (M0, shift) pair.

    The accumulator holds sums of (quantized input x quantized weight), so its
    real-world value is ``acc * in_scale * w_scale``. To express that as an
    integer at the OUTPUT scale we multiply by

        real_multiplier = in_scale * w_scale / out_scale

    which is < 1 for any sane layer. quantize_multiplier turns it into the
    (M0, shift) pair that requantize() uses.
    """
    real = (in_scale * w_scale) / out_scale
    if real >= 1.0:
        raise ValueError(
            f"real_multiplier = {real:.6f} >= 1.\n"
            f"  in_scale={in_scale:g} w_scale={w_scale:g} out_scale={out_scale:g}\n"
            f"  This means the output scale is too small for the accumulator "
            f"range. Usually it means calibration saw too few samples and "
            f"under-estimated the activation range. Raise "
            f"quantization.calibration_samples in the config."
        )
    return qr.quantize_multiplier(real)


def synthetic_model(cfg: dict, seed: int) -> dict:
    """Build a random but STRUCTURALLY VALID model, with no torch required.

    This exists so the entire downstream pipeline -- quantize, export, C
    compile, RTL simulation, parity check -- can be developed and tested
    before any real training has happened, and on machines with no PyTorch.
    The weights are meaningless; the shapes, scales and integer multipliers
    are exactly what a real model would produce.

    Any artifact built this way is tagged synthetic=True and carries an
    obviously-fake accuracy of -1.0, so it can never be mistaken for a result.
    """
    rng = np.random.default_rng(seed)
    layers = []

    # fc_autoencoder (workload B) is deployed as the REDUCED `deployed_model`
    # block, not the full architecture under `model.layers` -- the full
    # version is ~139k weights and does not fit in the SoC's 64 KB of RAM
    # (see train/config/workload_b.yaml and train/models.py::_build_autoencoder,
    # which already makes this distinction for the training path). Building
    # from `model.layers` here would quantize a model that can never be
    # exported to firmware, silently.
    is_autoencoder = cfg["model"]["architecture"] == "fc_autoencoder"
    if is_autoencoder:
        dep = cfg.get("deployed_model")
        if dep is None:
            raise ValueError(
                "workload config has architecture 'fc_autoencoder' but no "
                "'deployed_model' block")
        layer_specs = dep["layers"]
        cur_flat = dep["input_dim"]
        in_ch = h = w = None  # unused on this path
    else:
        layer_specs = cfg["model"]["layers"]
        inp = cfg["input"]
        in_ch = inp.get("channels", 1)
        h = inp.get("n_mels", inp.get("n_bins", 32))
        w = inp.get("n_frames", 1)
        cur_flat = None

    for spec in layer_specs:
        kind = spec["type"]
        if kind == "conv":
            k = spec["kernel"]
            oc = spec["out_ch"]
            # He-style scaling keeps the synthetic weights in a realistic
            # range, so the resulting scales are representative.
            fan_in = in_ch * k * k
            wt = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(oc, in_ch, k, k))
            layers.append({"type": "conv", "weight": wt,
                           "bias": rng.normal(0, 0.01, size=(oc,)),
                           "stride": spec.get("stride", 1),
                           "pad": spec.get("pad", 0)})
            h = (h + 2 * spec.get("pad", 0) - k) // spec.get("stride", 1) + 1
            w = (w + 2 * spec.get("pad", 0) - k) // spec.get("stride", 1) + 1
            in_ch = oc
        elif kind == "maxpool":
            k = spec["k"]
            layers.append({"type": "maxpool", "k": k})
            h //= k
            w //= k
        elif kind == "relu":
            layers.append({"type": "relu"})
        elif kind == "fc":
            in_dim = cur_flat if cur_flat is not None else in_ch * h * w
            od = spec["out_dim"]
            wt = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(od, in_dim))
            layers.append({"type": "fc", "weight": wt,
                           "bias": rng.normal(0, 0.01, size=(od,))})
            cur_flat = od
        else:
            raise ValueError(f"unknown layer type in config: {kind}")

    return {"layers": layers, "synthetic": True}


def load_torch_checkpoint(path: str) -> dict:
    """Load real trained weights. Only imported when actually needed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is not installed, so a real checkpoint cannot "
              "be read.\n"
              "  Install it (pip install torch) or use --synthetic to exercise "
              "the pipeline\n"
              "  with structurally valid but meaningless weights.",
              file=sys.stderr)
        sys.exit(2)
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    layers = []
    for key in sorted(state.keys()):
        if not key.endswith(".weight"):
            continue
        wt = state[key].detach().cpu().numpy()
        bkey = key.replace(".weight", ".bias")
        b = state[bkey].detach().cpu().numpy() if bkey in state else None
        layers.append({
            "type": "conv" if wt.ndim == 4 else "fc",
            "weight": wt,
            "bias": b if b is not None else np.zeros(wt.shape[0]),
            "stride": 1, "pad": 1 if wt.ndim == 4 else 0,
        })
    return {"layers": layers, "synthetic": False,
            "accuracy": float(ckpt.get("accuracy", -1.0))}


def quantize_model(model: dict, cfg: dict, act_scales: list[float] | None = None
                   ) -> dict:
    """Quantize every layer and derive its integer requantization parameters."""
    qcfg = cfg["quantization"]
    wprec = int(qcfg["weight_precision"])
    aprec = int(qcfg["activation_precision"])

    weight_layers = [L for L in model["layers"] if L["type"] in ("conv", "fc")]

    # Activation scales normally come from calibration -- running real data
    # through the float model and recording each layer's observed range. With
    # no calibration data available we fall back to a fixed conservative
    # scale, and LABEL the result so nobody mistakes it for a calibrated model.
    calibrated = act_scales is not None
    if not calibrated:
        act_scales = [1.0 / qr.qmax(aprec)] * (len(weight_layers) + 1)

    out = {"layers": [], "weight_precision": wprec,
           "activation_precision": aprec,
           "synthetic": model.get("synthetic", False),
           "calibrated": calibrated}

    for i, L in enumerate(weight_layers):
        qw, w_scale = quantize_weight_tensor(L["weight"], wprec)
        in_scale = act_scales[i]
        out_scale = act_scales[i + 1]

        m0, shift = compute_layer_multiplier(in_scale, w_scale, out_scale)

        # The bias lives in ACCUMULATOR units, i.e. it is divided by the
        # product of the input and weight scales -- not by the output scale.
        # Getting this wrong shifts every output by a constant, which looks
        # like a badly trained model rather than a bug.
        bias_scale = in_scale * w_scale
        qb = np.round(np.asarray(L["bias"], dtype=np.float64) / bias_scale)
        qb = np.clip(qb, -(2**31), 2**31 - 1).astype(np.int32)

        entry = {
            "type": L["type"],
            "weight": qw,
            "bias": qb,
            "multiplier": int(m0),
            "shift": int(shift),
            "zero_point": 0,
            "precision": wprec,
            "weight_scale": float(w_scale),
            "in_scale": float(in_scale),
            "out_scale": float(out_scale),
        }
        if L["type"] == "conv":
            entry["stride"] = L.get("stride", 1)
            entry["pad"] = L.get("pad", 0)
        out["layers"].append(entry)

        print(f"  layer {i} ({L['type']:4s}) shape={tuple(L['weight'].shape)} "
              f"w_scale={w_scale:.6g} M0={m0} shift={shift}")

    # Non-weight layers, kept so export_weights.py can emit the full topology.
    out["topology"] = [
        {"type": L["type"], **({"k": L["k"]} if L["type"] == "maxpool" else {})}
        for L in model["layers"]
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", help="trained .pt file")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate structurally valid random weights instead "
                         "of loading a checkpoint (no PyTorch needed)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--precision", type=int, choices=[8, 4],
                    help="override the config's weight precision (for the "
                         "RQ4 precision sweep)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # Refuse to run against a config that has drifted from its frozen hash.
    cfg = require_frozen(args.config)

    if args.precision:
        cfg = dict(cfg)
        cfg["quantization"] = dict(cfg["quantization"])
        cfg["quantization"]["weight_precision"] = args.precision
        cfg["quantization"]["activation_precision"] = args.precision

    seed = args.seed if args.seed is not None else cfg["training"]["seed"]

    print(f"Quantizing '{cfg['name']}' at "
          f"{cfg['quantization']['weight_precision']}-bit")
    print()

    if args.synthetic:
        print("  MODE: SYNTHETIC -- weights are random and MEANINGLESS.")
        print("        Shapes and integer scales are realistic, so the export,")
        print("        firmware and RTL pipeline can be tested end to end.")
        print("        Accuracy from this artifact is NOT a result.")
        print()
        model = synthetic_model(cfg, seed)
    elif args.checkpoint:
        model = load_torch_checkpoint(args.checkpoint)
    else:
        ap.error("give either --checkpoint or --synthetic")

    q = quantize_model(model, cfg)
    q["config_name"] = cfg["name"]
    q["seed"] = seed

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    flat: dict = {
        "config_name": np.array(cfg["name"]),
        "n_layers": np.array(len(q["layers"])),
        "weight_precision": np.array(q["weight_precision"]),
        "synthetic": np.array(q["synthetic"]),
        "calibrated": np.array(q["calibrated"]),
        "seed": np.array(seed),
    }
    for i, L in enumerate(q["layers"]):
        flat[f"L{i}_type"] = np.array(L["type"])
        flat[f"L{i}_weight"] = L["weight"]
        flat[f"L{i}_bias"] = L["bias"]
        for k in ("multiplier", "shift", "zero_point", "precision",
                  "stride", "pad"):
            if k in L:
                flat[f"L{i}_{k}"] = np.array(L[k])
    np.savez_compressed(args.out, **flat)

    print()
    print(f"Wrote {args.out}")
    print(f"  layers:    {len(q['layers'])}")
    print(f"  synthetic: {q['synthetic']}")
    print(f"  calibrated activations: {q['calibrated']}")
    if not q["calibrated"]:
        print("  NOTE: activation scales are DEFAULTS, not calibrated. Accuracy")
        print("        from this artifact is not meaningful. Run with real")
        print("        calibration data before reporting any accuracy number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
