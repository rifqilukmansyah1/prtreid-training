from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


ConfigDict = Dict[str, Any]


def load_yaml(path: Path) -> ConfigDict:
    """Load YAML file into a plain python dict."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Deep merge `override` into `base` and return a new dict."""
    out: ConfigDict = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config_bundle(base_config: Path, profile_config: Path | None) -> ConfigDict:
    """Load base config and apply optional profile override."""
    cfg = load_yaml(base_config)
    if profile_config is not None:
        cfg = deep_merge(cfg, load_yaml(profile_config))
    return cfg


def ensure_runtime_paths(cfg: ConfigDict, data_root: Path, output_dir: Path) -> ConfigDict:
    """Apply runtime paths to config dict."""
    cfg = dict(cfg)
    cfg.setdefault("data", {})
    cfg["data"]["root"] = str(data_root.resolve())
    cfg["data"]["save_dir"] = str(output_dir.resolve())
    return cfg
