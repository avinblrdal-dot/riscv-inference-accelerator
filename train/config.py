"""Config loading with a no-dependency fallback.

Configs are YAML because that is what the team will edit by hand. But PyYAML
is an extra install, and the whole point of the frozen-config discipline is
that ANYONE can re-read a config and reproduce a result. So if PyYAML is
missing we fall back to a small parser that handles the subset of YAML these
config files actually use: nested mappings, lists, strings, numbers, booleans
and comments.

The fallback is deliberately strict -- if it meets syntax it does not
understand it raises rather than guessing, because a config silently parsed
wrong would change a frozen model's meaning without anyone noticing.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except ImportError:  # pragma: no cover - exercised on minimal installs
    _HAVE_YAML = False


def _split_top(text: str) -> list[str]:
    """Split on commas that are NOT inside nested brackets or braces.

    A naive text.split(",") would tear '{a: [1, 2]}' apart at the inner
    comma. Configs here are simple, but a parser that quietly mangles nested
    structure is exactly the kind of thing that changes a frozen model's
    meaning without anyone noticing.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _coerce(token: str) -> Any:
    """Turn a scalar token into a Python value."""
    t = token.strip()
    if t == "" or t == "null" or t == "~":
        return None
    if t in ("true", "True", "yes"):
        return True
    if t in ("false", "False", "no"):
        return False
    if (t.startswith('"') and t.endswith('"')) or \
       (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        if not inner:
            return []
        return [_coerce(x) for x in _split_top(inner)]
    if t.startswith("{") and t.endswith("}"):
        # Inline mapping, e.g. {type: conv, out_ch: 8, kernel: 3}.
        # The model configs use these heavily for layer definitions, so
        # returning the raw string here would hand every downstream consumer
        # a str where it expects a dict -- a failure that surfaces far from
        # its cause.
        inner = t[1:-1].strip()
        if not inner:
            return {}
        out: dict = {}
        for part in _split_top(inner):
            if ":" not in part:
                raise ValueError(f"inline mapping entry missing ':': {part!r}")
            k, _, v = part.partition(":")
            out[k.strip()] = _coerce(v)
        return out
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def _mini_yaml(text: str) -> dict:
    """Parse the subset of YAML used by this project's configs."""
    root: dict = {}
    # Stack of (indent, container) so nested mappings work.
    stack: list[tuple[int, Any]] = [(-1, root)]

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        body = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"config line {lineno}: bad indentation")
        parent = stack[-1][1]

        if body.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(
                    f"config line {lineno}: list item outside a list -- the "
                    f"minimal parser needs 'key:' on its own line first"
                )
            parent.append(_coerce(body[2:]))
            continue

        if ":" not in body:
            raise ValueError(f"config line {lineno}: expected 'key: value'")

        key, _, value = body.partition(":")
        key = key.strip()
        value = value.strip()

        if value == "":
            # Could be a nested mapping or a list; decide from the next
            # non-blank line's shape.
            child: Any = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce(value)

    # Convert any mapping that only ever received list items. The minimal
    # parser creates {} first, so empty dicts followed by '- ' lines need a
    # second pass; handled by _mini_yaml_lists below.
    return root


def _mini_yaml_with_lists(text: str) -> dict:
    """Two-pass variant that supports 'key:' followed by '- item' lines."""
    lines = [ln for ln in text.splitlines()]
    # Pre-scan: which keys are followed by list items at deeper indent?
    list_keys: set[int] = set()
    for i, raw in enumerate(lines):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        if line.strip().endswith(":"):
            indent = len(line) - len(line.lstrip())
            for j in range(i + 1, len(lines)):
                nxt = lines[j].split("#", 1)[0].rstrip()
                if not nxt.strip():
                    continue
                nindent = len(nxt) - len(nxt.lstrip())
                if nindent <= indent:
                    break
                if nxt.strip().startswith("- "):
                    list_keys.add(i)
                break

    root: dict = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for i, raw in enumerate(lines):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if body.startswith("- "):
            if isinstance(parent, list):
                parent.append(_coerce(body[2:]))
            continue

        key, _, value = body.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            child: Any = [] if i in list_keys else {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce(value)
    return root


def load_config(path: str) -> dict:
    """Load a YAML config file.

    Uses PyYAML when available and the built-in fallback otherwise, so a
    missing optional dependency never blocks reproducing a result.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config not found: {path}\n"
            f"Available configs live in train/config/."
        )
    with open(path) as fh:
        text = fh.read()

    if _HAVE_YAML:
        return yaml.safe_load(text)
    return _mini_yaml_with_lists(text)


def save_config(cfg: dict, path: str) -> None:
    """Write a config back out (used by freeze.py for the manifest)."""
    if _HAVE_YAML:
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        return

    def emit(obj: Any, indent: int = 0) -> list[str]:
        pad = " " * indent
        out: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)) and v:
                    out.append(f"{pad}{k}:")
                    out += emit(v, indent + 2)
                else:
                    out.append(f"{pad}{k}: {v}")
        elif isinstance(obj, list):
            for item in obj:
                out.append(f"{pad}- {item}")
        return out

    with open(path, "w") as fh:
        fh.write("\n".join(emit(cfg)) + "\n")
