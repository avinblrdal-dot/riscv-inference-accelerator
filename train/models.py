#!/usr/bin/env python3
"""PyTorch model definitions for both workloads.

The architectures are driven entirely by the frozen YAML configs, so the
config is the single source of truth for what a model IS. Hardcoding a shape
here that disagreed with the config would make the frozen hash a lie.

DESIGN CONSTRAINT THAT SHAPES EVERYTHING
----------------------------------------
The model must fit in the SoC's 64 KB of on-chip RAM (rtl/soc_top.v
MEM_WORDS = 16384 words), sharing it with activations, the stack and the
program itself. As int8, that caps weights at roughly 15k parameters for
workload A. Accuracy ambition is bounded by that budget, not the other way
round -- which is the honest situation for a battery-powered sensor node and
is worth stating in the paper rather than hiding.

PyTorch is imported lazily so that merely importing this module does not
require torch. quant_ref.py, verify_parity.py and the whole export path stay
torch-free by design.
"""

from __future__ import annotations

import sys
from typing import Any


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        print(
            "ERROR: PyTorch is required to build or train a model.\n"
            "  pip install torch     (or: pip install -r train/requirements.txt)\n"
            "\n"
            "  You do NOT need PyTorch to run the bit-exactness harness, the\n"
            "  RTL simulations, or the export pipeline -- use\n"
            "  'python3 train/quantize.py --synthetic' for those.",
            file=sys.stderr)
        sys.exit(2)


def build_model(cfg: dict) -> Any:
    """Construct the torch model described by a frozen config."""
    torch, nn = _require_torch()

    arch = cfg["model"]["architecture"]
    if arch == "small_cnn":
        return _build_cnn(cfg, nn)
    if arch == "fc_autoencoder":
        return _build_autoencoder(cfg, nn)
    raise ValueError(f"unknown architecture '{arch}' in config {cfg['name']}")


def _build_cnn(cfg: dict, nn) -> Any:
    """Workload A: a small convolutional classifier.

    No batch normalisation anywhere. That is deliberate: BN would have to be
    folded into the preceding convolution before quantization, and an
    unfolded BN is a classic source of accuracy loss that gets misdiagnosed
    as a quantization problem. Omitting it keeps the quantized graph a
    faithful one-to-one match of the float graph, which is what makes
    bit-exactness with the C and RTL achievable.
    """
    inp = cfg["input"]
    in_ch = inp.get("channels", 1)
    h = inp["n_mels"]
    w = inp["n_frames"]

    layers = []
    cur_ch = in_ch
    for spec in cfg["model"]["layers"]:
        t = spec["type"]
        if t == "conv":
            layers.append(nn.Conv2d(cur_ch, spec["out_ch"], spec["kernel"],
                                    stride=spec.get("stride", 1),
                                    padding=spec.get("pad", 0),
                                    bias=True))
            cur_ch = spec["out_ch"]
            h = (h + 2*spec.get("pad", 0) - spec["kernel"]) // spec.get("stride", 1) + 1
            w = (w + 2*spec.get("pad", 0) - spec["kernel"]) // spec.get("stride", 1) + 1
        elif t == "relu":
            layers.append(nn.ReLU())
        elif t == "maxpool":
            layers.append(nn.MaxPool2d(spec["k"]))
            h //= spec["k"]
            w //= spec["k"]
        elif t == "fc":
            layers.append(nn.Flatten())
            layers.append(nn.Linear(cur_ch * h * w, spec["out_dim"], bias=True))
            cur_ch = spec["out_dim"]
            h = w = 1
        else:
            raise ValueError(f"unknown layer type '{t}'")

    model = nn.Sequential(*layers)

    n_params = sum(p.numel() for p in model.parameters())
    budget = cfg["model"].get("max_parameters")
    if budget and n_params > budget:
        raise ValueError(
            f"model has {n_params} parameters, over the {budget} budget in the "
            f"config.\n  It will not fit in the SoC's 64 KB of RAM. Reduce a "
            f"channel count, or raise MEM_WORDS in rtl/soc_top.v AND the RAM "
            f"length in sw/link.ld together."
        )
    print(f"  built {cfg['model']['architecture']}: {n_params} parameters "
          f"(~{n_params/1024:.1f} KB as int8, budget {budget})")
    return model


def _build_autoencoder(cfg: dict, nn) -> Any:
    """Workload B: a fully-connected autoencoder. No convolutions at all.

    The structural difference from workload A is the entire point -- see the
    header of train/config/workload_b.yaml. Using the `deployed_model` block
    keeps the on-chip variant inside the memory budget.
    """
    dep = cfg.get("deployed_model")
    if dep is None:
        raise ValueError("workload_b config needs a 'deployed_model' block")

    in_dim = dep["input_dim"]
    layers = []
    cur = in_dim
    for spec in dep["layers"]:
        if spec["type"] == "fc":
            layers.append(nn.Linear(cur, spec["out_dim"], bias=True))
            cur = spec["out_dim"]
        elif spec["type"] == "relu":
            layers.append(nn.ReLU())
        else:
            raise ValueError(f"unknown layer type '{spec['type']}'")

    if cur != in_dim:
        raise ValueError(
            f"an autoencoder must reconstruct its input: final layer outputs "
            f"{cur} but the input is {in_dim}."
        )

    model = nn.Sequential(*layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  built fc_autoencoder: {n_params} parameters "
          f"(~{n_params/1024:.1f} KB as int8)")
    return model


def count_macs(cfg: dict) -> int:
    """Analytic MAC count for one forward pass.

    Computed from the config alone, with no torch, so it is available
    everywhere. This is the denominator for energy-per-operation (pJ/MAC),
    which is the unit that makes results transferable beyond these two models
    -- and therefore the unit the paper should lead with.
    """
    arch = cfg["model"]["architecture"]
    total = 0

    if arch == "small_cnn":
        inp = cfg["input"]
        ch = inp.get("channels", 1)
        h, w = inp["n_mels"], inp["n_frames"]
        for spec in cfg["model"]["layers"]:
            t = spec["type"]
            if t == "conv":
                k, oc = spec["kernel"], spec["out_ch"]
                st, pad = spec.get("stride", 1), spec.get("pad", 0)
                oh = (h + 2*pad - k)//st + 1
                ow = (w + 2*pad - k)//st + 1
                total += oc * oh * ow * ch * k * k
                ch, h, w = oc, oh, ow
            elif t == "maxpool":
                h //= spec["k"]; w //= spec["k"]
            elif t == "fc":
                total += ch * h * w * spec["out_dim"]
                ch, h, w = spec["out_dim"], 1, 1
    elif arch == "fc_autoencoder":
        dep = cfg["deployed_model"]
        cur = dep["input_dim"]
        for spec in dep["layers"]:
            if spec["type"] == "fc":
                total += cur * spec["out_dim"]
                cur = spec["out_dim"]
    return total


if __name__ == "__main__":
    # Reports MAC counts without needing torch -- useful for sizing the
    # experiment before any training has happened.
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import load_config

    for name in ("workload_a", "workload_b"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "config", f"{name}.yaml")
        cfg = load_config(path)
        print(f"{cfg['name']}: {count_macs(cfg):,} MACs per inference")
