#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (src overrides dst)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping at top-level: {path}")
    return data


def to_cli_args(cfg: Dict[str, Any]) -> List[str]:
    """
    Convert flat config dict to CLI args:
    - key: value -> --key value
    - bool True -> --key
    - bool False/None -> omitted
    - list -> repeated --key item
    """
    args: List[str] = []
    for k, v in cfg.items():
        if v is None:
            continue

        key = f"--{k}"

        if isinstance(v, bool):
            if v:
                args.append(key)
            continue

        if isinstance(v, (list, tuple)):
            for item in v:
                if item is None:
                    continue
                args.extend([key, str(item)])
            continue

        # numbers/strings/etc
        args.extend([key, str(v)])

    return args


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        action="append",
        required=True,
        help="YAML config file(s). Later ones override earlier ones.",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print resolved config and args, do not run.")
    ap.add_argument(
        "--print_args",
        action="store_true",
        help="Print generated CLI args only (shell-escaped) and exit.",
    )
    ap.add_argument(
        "--train_py",
        default="scripts/train.py",
        help="Path to train.py (default: scripts/train.py)",
    )
    ap.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args appended after '--' (e.g., -- --max_epochs 10)",
    )
    args = ap.parse_args()

    # Merge configs
    cfg: Dict[str, Any] = {}
    for cpath in args.config:
        cfg = deep_update(cfg, load_yaml(cpath))

    # Optional: environment overrides (handy in containers)
    # If LOG_ROOT is set, map it to checkpoint_root unless user explicitly sets it.
    if "checkpoint_root" not in cfg and os.environ.get("CHECKPOINT_ROOT"):
        cfg["checkpoint_root"] = os.environ["CHECKPOINT_ROOT"]

    # Ensure run_name exists if user wants it fixed externally
    # (If you always set run_name in YAML or wrapper, nothing happens here.)
    # If RUN_ID is provided, use it.
    if cfg.get("run_name") in (None, "", "auto") and os.environ.get("RUN_ID"):
        cfg["run_name"] = os.environ["RUN_ID"]

    cli = to_cli_args(cfg)

    # Extra args handling: allow caller to write: run_from_yaml.py --config ... -- --max_epochs 10
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]
    cli = cli + extra

    if args.print_args:
        print(" ".join(shlex.quote(x) for x in cli))
        return

    if args.dry_run:
        import json
        print("[resolved config]")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        print("\n[cli args]")
        print(" ".join(shlex.quote(x) for x in cli))
        return

    cmd = ["python", args.train_py] + cli
    print("[run]", " ".join(shlex.quote(x) for x in cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()