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


def load_game(game: str | None) -> dict[str, Any]:
    """Game profile from config/games/<game>.yaml ({} if none)."""
    if not game:
        return {}
    path = ROOT / "config" / "games" / f"{game}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no game profile at {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def channel_settings(settings: dict, channel: dict) -> dict:
    """Global settings < the channel's game profile < the channel's own overrides."""
    out = _deep_merge(settings, load_game(channel.get("game")))
    keys = ("detector", "clip", "ranking", "reel", "signals", "audio", "speech")
    overrides = {k: channel[k] for k in keys if k in channel}
    out = _deep_merge(out, overrides)
    out.setdefault("moments", [])
    out.setdefault("game_name", channel.get("game") or "")
    out["streamer"] = channel.get("streamer") or channel["name"].lstrip("@")
    return out
