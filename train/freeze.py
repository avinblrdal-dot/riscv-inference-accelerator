#!/usr/bin/env python3
"""Lock a config so a model becomes an immutable, citable artifact.

WHY FREEZING MATTERS
--------------------
Every number this project reports is "the energy of THIS model at THIS
precision". If the model quietly changes -- someone retrains with a different
seed, tweaks a layer, adjusts the mel bin count -- then measurements taken
weeks apart are no longer comparable, and nobody notices, because the numbers
still look plausible.

Freezing makes that failure impossible to have by accident:

  1. ``freeze.py --config X`` hashes the config and writes a manifest.
  2. Any script that loads a frozen config re-hashes it and STOPS if it has
     drifted.
  3. ``freeze.py`` refuses to overwrite an existing manifest without
     ``--force``, and explains why.

The hash covers the config CONTENT, not the filename or timestamps, so
reformatting whitespace does not spuriously invalidate a model, but changing
a single hyperparameter does.

Usage:
    python3 train/freeze.py --config train/config/workload_a.yaml
    python3 train/freeze.py --config ... --verify      # check, do not write
    python3 train/freeze.py --list
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from config import load_config  # noqa: E402

MANIFEST_PATH = os.path.join(HERE, "frozen_manifest.json")


def canonical_bytes(cfg: dict) -> bytes:
    """Serialize a config so equal configs always hash identically.

    sort_keys makes key order irrelevant, and the separators remove
    whitespace variation. Without this, two logically identical configs could
    hash differently and trigger spurious "model changed" errors -- which
    would train the team to ignore the warning, defeating the whole point.
    """
    return json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(canonical_bytes(cfg)).hexdigest()


def git_sha() -> str:
    """Record the code version alongside the config version."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                   capture_output=True, text=True, timeout=10)
            return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return "unknown"


def load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


def verify_frozen(config_path: str) -> tuple[bool, str]:
    """Check a config against the manifest.

    Returns (ok, message). Callers in the training and export pipeline should
    treat ok=False as fatal.
    """
    cfg = load_config(config_path)
    name = cfg.get("name")
    if name is None:
        return False, f"{config_path} has no 'name' field"

    manifest = load_manifest()
    entry = manifest.get(name)
    if entry is None:
        return False, (
            f"'{name}' is not frozen yet.\n"
            f"  Freeze it with:  python3 train/freeze.py --config {config_path}"
        )

    current = config_hash(cfg)
    if current != entry["config_sha256"]:
        return False, (
            f"FROZEN MODEL HAS CHANGED -- '{name}'\n"
            f"  frozen hash:  {entry['config_sha256']}\n"
            f"  current hash: {current}\n"
            f"  frozen on:    {entry['frozen_utc']}\n"
            f"\n"
            f"  Every measurement taken against this model is now suspect.\n"
            f"  Either revert your edits to {config_path}, or -- if the change\n"
            f"  is intentional -- create a NEW config with a new 'name' rather\n"
            f"  than modifying this one. Do not simply re-freeze: that would\n"
            f"  silently invalidate results already in the notebook."
        )
    return True, f"'{name}' matches its frozen hash ({current[:12]}...)"


def require_frozen(config_path: str) -> dict:
    """Load a config, refusing to proceed unless it is frozen and unchanged.

    Import this from any script that produces a measurable artifact.
    """
    ok, msg = verify_frozen(config_path)
    if not ok:
        print("=" * 70, file=sys.stderr)
        print(" REFUSING TO PROCEED", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(msg, file=sys.stderr)
        sys.exit(2)
    return load_config(config_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to the config to freeze or verify")
    ap.add_argument("--verify", action="store_true",
                    help="only check; do not write the manifest")
    ap.add_argument("--list", action="store_true",
                    help="show every frozen model")
    ap.add_argument("--force", action="store_true",
                    help="re-freeze a config that is already frozen "
                         "(invalidates prior measurements -- say so in the "
                         "lab notebook)")
    args = ap.parse_args()

    if args.list:
        manifest = load_manifest()
        if not manifest:
            print("Nothing frozen yet.")
            return 0
        print(f"{'name':<40} {'frozen (UTC)':<22} hash")
        print("-" * 92)
        for name, e in sorted(manifest.items()):
            print(f"{name:<40} {e['frozen_utc']:<22} {e['config_sha256'][:16]}...")
        return 0

    if not args.config:
        ap.error("--config is required unless --list is given")

    cfg = load_config(args.config)
    name = cfg.get("name")
    if not name:
        print(f"ERROR: {args.config} has no 'name' field.", file=sys.stderr)
        return 2

    if args.verify:
        ok, msg = verify_frozen(args.config)
        print(msg)
        return 0 if ok else 1

    manifest = load_manifest()
    new_hash = config_hash(cfg)

    if name in manifest and not args.force:
        existing = manifest[name]
        if existing["config_sha256"] == new_hash:
            print(f"'{name}' is already frozen and unchanged. Nothing to do.")
            return 0
        print("=" * 70, file=sys.stderr)
        print(" REFUSING TO RE-FREEZE", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"  '{name}' was frozen on {existing['frozen_utc']} with hash", file=sys.stderr)
        print(f"  {existing['config_sha256']}", file=sys.stderr)
        print(f"  but {args.config} now hashes to", file=sys.stderr)
        print(f"  {new_hash}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Re-freezing would silently invalidate every measurement", file=sys.stderr)
        print("  already recorded against the old model.", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Preferred fix: create a NEW config with a new 'name'.", file=sys.stderr)
        print("  If you are certain, re-run with --force AND write a dated", file=sys.stderr)
        print("  entry in notebook/NOTEBOOK.md explaining what changed and", file=sys.stderr)
        print("  which results are now void.", file=sys.stderr)
        return 3

    manifest[name] = {
        "config_path": os.path.relpath(args.config, ROOT),
        "config_sha256": new_hash,
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc)
                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "seed": cfg.get("training", {}).get("seed"),
    }
    save_manifest(manifest)

    print(f"Froze '{name}'")
    print(f"  config: {args.config}")
    print(f"  sha256: {new_hash}")
    print(f"  git:    {manifest[name]['git_sha']}")
    print()
    print("This config is now immutable. Scripts that produce measurable")
    print("artifacts will verify this hash and refuse to run if it changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
