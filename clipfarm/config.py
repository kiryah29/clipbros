from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or ROOT / "config" / "settings.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_channels(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path or ROOT / "config" / "channels.yaml")
    with open(path) as f:
        return yaml.safe_load(f).get("channels", [])


def channel_settings(settings: dict, channel: dict) -> dict:
    """Global settings with the channel's per-section overrides applied."""
    overrides = {k: channel[k] for k in ("detector", "clip", "ranking", "reel") if k in channel}
    return _deep_merge(settings, overrides)
