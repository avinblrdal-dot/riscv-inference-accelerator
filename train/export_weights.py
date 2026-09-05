#!/usr/bin/env python3
"""Export a quantized model to a C header and matching golden vectors.

WHAT THIS PRODUCES
------------------
    sw/models/model_weights.h    weights, biases, topology and test input,
                                 as C arrays the firmware #includes
    sim/golden/<name>.npz        the reference activations and output that
                                 verify_parity.py compares C and RTL against
    sim/golden/<name>.json       human-readable metadata

WHY THE GOLDEN VECTORS MATTER MORE THAN THE HEADER
--------------------------------------------------
The header is just data. The golden vectors are what make a hardware bug
DETECTABLE. They record, for a specific known input, exactly what every layer
should produce, computed by the Python reference. When the C or the RTL
disagrees, verify_parity.py can say "layer 2, output index 47, expected -13,
got -14" instead of "the accuracy is a bit lower than expected".

The test input is deterministic, derived from the model's seed, so anyone can
regenerate byte-identical vectors from a clean clone.

Usage:
    python3 train/export_weights.py --quantized train/runs/workload_a/quantized.npz \
        --config train/config/workload_a.yaml
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import quant_ref as qr              # noqa: E402
from freeze import require_frozen, git_sha  # noqa: E402


def c_array(name: str, data: np.ndarray, ctype: str, per_line: int = 12) -> str:
    """Emit a flat C array. Weights are `const` so they stay in .rodata."""
    flat = np.asarray(data).ravel()
    lines = [f"static const {ctype} {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per_line):
        chunk = ", ".join(str(int(v)) for v in flat[i:i + per_line])
        lines.append(f"    {chunk},")
    lines.append("};")
    return "\n".join(lines)


def run_reference(layers: list[dict], topology: list[str],
                  x: np.ndarray, in_shape: tuple) -> tuple[np.ndarray, list]:
    """Run the Python reference forward pass, capturing every intermediate.

    The captures are the golden vectors. Capturing per LAYER rather than only
    the final output is deliberate: if only the output were checked, a bug in
    layer 1 that happens to cancel out would be missed, and a bug in layer 3
    would give no clue where to look.
    """
    cur = x.reshape(in_shape)
    captures = []
    wi = 0

    for kind in topology:
        if kind == "conv":
            L = layers[wi]
            acc = qr.conv2d_int(cur, L["weight"], L["bias"],
                                stride=L.get("stride", 1), pad=L.get("pad", 0))
            cur = qr.requantize(acc, L["multiplier"], L["shift"],
                                L["zero_point"], L["precision"]).astype(np.int8)
            captures.append(("conv", cur.copy()))
            wi += 1
        elif kind == "relu":
            cur = qr.relu_int(cur, 0).astype(np.int8)
            captures.append(("relu", cur.copy()))
        elif kind == "maxpool":
            cur = qr.maxpool2d_int(cur, k=2).astype(np.int8)
            captures.append(("maxpool", cur.copy()))
        elif kind == "fc":
            L = layers[wi]
            flat = cur.reshape(-1)
            acc = qr.linear_int(flat, L["weight"].reshape(L["weight"].shape[0], -1),
                                L["bias"])
            cur = qr.requantize(acc, L["multiplier"], L["shift"],
                                L["zero_point"], L["precision"]).astype(np.int8)
            captures.append(("fc", cur.copy()))
            wi += 1
    return cur, captures


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantized", required=True, help="the .npz from quantize.py")
    ap.add_argument("--config", required=True)
    ap.add_argument("--header-out", default=None,
                    help="default sw/models/model_weights.h")
    ap.add_argument("--golden-dir", default=None,
                    help="default sim/golden/")
    args = ap.parse_args()

    cfg = require_frozen(args.config)
    z = np.load(args.quantized, allow_pickle=False)

    n_layers = int(z["n_layers"])
    synthetic = bool(z["synthetic"])
    seed = int(z["seed"])

    layers = []
    for i in range(n_layers):
        L = {
            "type": str(z[f"L{i}_type"]),
            "weight": z[f"L{i}_weight"],
            "bias": z[f"L{i}_bias"],
            "multiplier": int(z[f"L{i}_multiplier"]),
            "shift": int(z[f"L{i}_shift"]),
            "zero_point": int(z[f"L{i}_zero_point"]),
            "precision": int(z[f"L{i}_precision"]),
        }
        if f"L{i}_stride" in z:
            L["stride"] = int(z[f"L{i}_stride"])
            L["pad"] = int(z[f"L{i}_pad"])
        layers.append(L)

    # fc_autoencoder (workload B) is deployed as the REDUCED `deployed_model`
    # block -- see the matching comment in quantize.py::synthetic_model. Using
    # `cfg["model"]["layers"]` here would build a topology table and an input
    # shape for the ~139k-weight full model, which does not match the ~8.8k
    # weights actually being exported and does not fit in the SoC's RAM.
    is_autoencoder = cfg["model"]["architecture"] == "fc_autoencoder"
    if is_autoencoder:
        dep = cfg["deployed_model"]
        layer_specs = dep["layers"]
        in_ch, in_h, in_w = 1, dep["input_dim"], 1
    else:
        layer_specs = cfg["model"]["layers"]
        inp = cfg["input"]
        in_ch = inp.get("channels", 1)
        in_h = inp.get("n_mels", inp.get("n_bins", 32))
        in_w = inp.get("n_frames", 1)

    topology = [s["type"] for s in layer_specs]
    in_shape = (in_ch, in_h, in_w)

    # Deterministic test input from the model seed, so the golden vectors are
    # reproducible from a clean clone by anyone.
    rng = np.random.default_rng(seed)
    test_input = rng.integers(-128, 128, size=in_shape).astype(np.int8)

    print(f"Running the Python reference over a {in_shape} input...")
    out, captures = run_reference(layers, topology, test_input, in_shape)
    predicted = int(np.argmax(out.ravel()))
    print(f"  reference output: {out.ravel().tolist()}")
    recon_mae = None
    if is_autoencoder:
        # argmax over a reconstruction is not a meaningful "class" -- an
        # autoencoder's output IS the answer (the reconstructed vector), not
        # an index into it. Kept only as MODEL_EXPECTED_CLASS below so the
        # existing generic self-check plumbing still has a scalar to compare,
        # but sw/src/main.c must not report it as a classification result.
        # MAE, not MSE: matches detection.metric in workload_b.yaml, and
        # squaring in int32 risks overflow on a 512-dim vector (see the
        # config's own comment on this choice).
        recon_mae = int(np.mean(np.abs(out.ravel().astype(np.int32)
                                       - test_input.ravel().astype(np.int32))))
        print(f"  reconstruction MAE vs input (synthetic, meaningless): {recon_mae}")
    else:
        print(f"  predicted class:  {predicted}")

    max_tensor = max(int(np.prod(in_shape)),
                     max(int(c[1].size) for c in captures))

    # ---- the C header ----------------------------------------------------
    header_path = args.header_out or os.path.join(ROOT, "sw", "models",
                                                  "model_weights.h")
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    model_hash = hashlib.sha256(
        b"".join(np.asarray(L["weight"]).tobytes() for L in layers)
    ).hexdigest()

    parts: list[str] = []
    parts.append(f"""/* =========================================================================
 * model_weights.h -- GENERATED FILE, DO NOT EDIT
 * =========================================================================
 *
 * Generated by train/export_weights.py
 *   config     : {cfg['name']}
 *   generated  : {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
 *   git        : {git_sha()}
 *   seed       : {seed}
 *   weight hash: {model_hash[:32]}
 *   SYNTHETIC  : {str(synthetic).upper()}
 *
 * {'*** THESE WEIGHTS ARE RANDOM AND MEANINGLESS. ***' if synthetic else 'Trained weights.'}
 * {'They exist so the firmware and RTL pipeline can be tested before' if synthetic else ''}
 * {'training exists. Any accuracy measured with them is NOT a result.' if synthetic else ''}
 *
 * Regenerate with:  make weights
 * ========================================================================= */

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>
#include "nn.h"

#define MODEL_NAME        "{cfg['name']}"
#define MODEL_HASH        "{model_hash[:16]}"
#define MODEL_SYNTHETIC   {1 if synthetic else 0}
#define MODEL_SEED        {seed}

#define MODEL_INPUT_CH    {in_ch}
#define MODEL_INPUT_H     {in_h}
#define MODEL_INPUT_W     {in_w}
#define MODEL_NUM_CLASSES {int(out.size)}
#define MODEL_MAX_TENSOR  {max_tensor}

/* Layer kinds, mirrored by the switch in sw/src/main.c. */
typedef enum {{
    LAYER_CONV = 0,
    LAYER_RELU = 1,
    LAYER_MAXPOOL = 2,
    LAYER_FC = 3
}} layer_kind_t;

typedef struct {{
    layer_kind_t kind;
    nn_conv_t    conv;
    nn_fc_t      fc;
    int32_t      n_elems;   /* for relu */
    int32_t      pool_ch;   /* for maxpool */
    int32_t      pool_k;
}} model_layer_t;
""")

    for i, L in enumerate(layers):
        parts.append(c_array(f"w{i}", L["weight"], "int8_t"))
        parts.append(c_array(f"b{i}", L["bias"], "int32_t"))

    parts.append(c_array("model_test_input_data", test_input, "int8_t"))
    parts.append("static const int8_t *const model_test_input = "
                 "model_test_input_data;")

    # The reference output, so the firmware can self-check on the board with
    # no host in the loop.
    parts.append(c_array("model_expected_output", out.ravel(), "int8_t"))
    parts.append(f"#define MODEL_EXPECTED_CLASS {predicted}")
    # MODEL_TASK tells sw/src/main.c whether the final tensor is a class
    # score vector (argmax it) or a reconstruction (diff it against the
    # input). Getting this wrong would make main.c report a meaningless
    # "class" for the autoencoder -- exactly the kind of silent nonsense
    # result this project's bit-exactness discipline exists to prevent.
    parts.append(f"#define MODEL_TASK_CLASSIFY   0")
    parts.append(f"#define MODEL_TASK_RECONSTRUCT 1")
    parts.append(f"#define MODEL_TASK "
                 f"{'MODEL_TASK_RECONSTRUCT' if is_autoencoder else 'MODEL_TASK_CLASSIFY'}")
    # Bit-exact reference for the reconstruction path, analogous to
    # MODEL_EXPECTED_CLASS on the classification path. Firmware and RTL
    # simulation are deterministic integer arithmetic, so this MAE is an
    # exact value to check against -- not an estimate -- the same way a
    # golden class is.
    if recon_mae is not None:
        parts.append(f"#define MODEL_EXPECTED_RECONSTRUCTION_MAE {recon_mae}")

    # Topology table.
    entries = []
    wi = 0
    spatial = [in_ch, in_h, in_w]
    for spec in layer_specs:
        t = spec["type"]
        if t == "conv":
            L = layers[wi]
            oc, ic, kh, kw = L["weight"].shape
            entries.append(
                f"    {{ LAYER_CONV, {{ w{wi}, b{wi}, {oc}, {ic}, {kh}, {kw}, "
                f"{L.get('stride',1)}, {L.get('pad',0)}, "
                f"{{ {L['multiplier']}, {L['shift']}, {L['zero_point']}, "
                f"{L['precision']} }} }}, {{0,0,0,0,{{0,0,0,0}}}}, 0, 0, 0 }},")
            spatial = [oc,
                       (spatial[1] + 2*L.get('pad',0) - kh)//L.get('stride',1) + 1,
                       (spatial[2] + 2*L.get('pad',0) - kw)//L.get('stride',1) + 1]
            wi += 1
        elif t == "relu":
            n = spatial[0] * spatial[1] * spatial[2]
            entries.append(
                f"    {{ LAYER_RELU, {{0,0,0,0,0,0,0,0,{{0,0,0,0}}}}, "
                f"{{0,0,0,0,{{0,0,0,0}}}}, {n}, 0, 0 }},")
        elif t == "maxpool":
            k = spec["k"]
            entries.append(
                f"    {{ LAYER_MAXPOOL, {{0,0,0,0,0,0,0,0,{{0,0,0,0}}}}, "
                f"{{0,0,0,0,{{0,0,0,0}}}}, 0, {spatial[0]}, {k} }},")
            spatial = [spatial[0], spatial[1]//k, spatial[2]//k]
        elif t == "fc":
            L = layers[wi]
            od = L["weight"].shape[0]
            idim = int(np.prod(L["weight"].shape[1:]))
            entries.append(
                f"    {{ LAYER_FC, {{0,0,0,0,0,0,0,0,{{0,0,0,0}}}}, "
                f"{{ w{wi}, b{wi}, {od}, {idim}, "
                f"{{ {L['multiplier']}, {L['shift']}, {L['zero_point']}, "
                f"{L['precision']} }} }}, 0, 0, 0 }},")
            spatial = [od, 1, 1]
            wi += 1

    parts.append(f"#define MODEL_NUM_LAYERS {len(entries)}\n"
                 "static const model_layer_t model_layers[MODEL_NUM_LAYERS] = {\n"
                 + "\n".join(entries) + "\n};")
    parts.append("#endif /* MODEL_WEIGHTS_H */")

    with open(header_path, "w") as fh:
        fh.write("\n\n".join(parts) + "\n")
    print(f"Wrote {header_path}")

    # ---- golden vectors ---------------------------------------------------
    golden_dir = args.golden_dir or os.path.join(ROOT, "sim", "golden")
    os.makedirs(golden_dir, exist_ok=True)

    golden = {"input": test_input, "output": out.ravel(),
              "predicted_class": np.array(predicted)}
    for i, (kind, arr) in enumerate(captures):
        golden[f"layer{i}_{kind}"] = arr

    npz_path = os.path.join(golden_dir, f"{cfg['name']}.npz")
    np.savez_compressed(npz_path, **golden)

    meta = {
        "config_name": cfg["name"],
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "seed": seed,
        "synthetic": synthetic,
        "weight_sha256": model_hash,
        "input_shape": list(in_shape),
        "num_classes": int(out.size),
        "predicted_class": predicted,
        "task": "reconstruct" if is_autoencoder else "classify",
        "expected_reconstruction_mae": recon_mae,
        "layers": [
            {"index": i, "type": L["type"],
             "weight_shape": list(np.asarray(L["weight"]).shape),
             "multiplier": L["multiplier"], "shift": L["shift"],
             "precision": L["precision"]}
            for i, L in enumerate(layers)
        ],
        "note": ("SYNTHETIC weights -- not a trained model. Any accuracy "
                 "computed from this is meaningless."
                 if synthetic else "Trained model."),
    }
    json_path = os.path.join(golden_dir, f"{cfg['name']}.json")
    with open(json_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote {npz_path}")
    print(f"Wrote {json_path}")
    print(f"  captured {len(captures)} intermediate tensors as golden vectors")
    if synthetic:
        print()
        print("  REMINDER: this artifact is SYNTHETIC. It proves the pipeline")
        print("  works; it does not measure anything about real data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
