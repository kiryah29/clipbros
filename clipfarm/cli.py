from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .config import ROOT, channel_settings, load_channels, load_settings
from .engine import craft_clip, run_demo, run_live
from .queue import ClipQueue


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _print_queue(q: ClipQueue, status: str | None, limit: int) -> None:
    clips = q.ranked(status, limit)
    if not clips:
        print("Queue is empty.")
        return
    print(f"{'rank':>4} {'id':>4} {'score':>7}  {'status':<9} {'channel':<18} why")
    for i, c in enumerate(clips, 1):
        print(f"{i:>4} {c.id:>4} {c.score:>7.2f}  {c.status:<9} {c.channel:<18} {c.receipt.get('why', '')}")
        if c.reel_path:
            print(f"{'':>17}reel: {c.reel_path}")


def main(argv: list[str] | None = None) -> None:
    _load_env()
    for stream in (sys.stdout, sys.stderr):  # emotes/emoji on Windows consoles
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="clipfarm", description="Chat-velocity live clipper")
    p.add_argument("--settings", default=None)
    p.add_argument("--channels", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the full pipeline on simulated chat (no API keys)")
    d.add_argument("--minutes", type=float, default=20)
    d.add_argument("--no-render", action="store_true", help="detect + queue only, skip ffmpeg")
    d.add_argument("--seed", type=int, default=7)

    sub.add_parser("run", help="monitor enabled live channels")

    w = sub.add_parser("watch", help="chat-only test: show live chat pace and spikes, no recording")
    w.add_argument("platform", choices=["twitch", "youtube", "kick"])
    w.add_argument("channel")

    qp = sub.add_parser("queue", help="show the ranked clip queue")
    qp.add_argument("--status", default=None)
    qp.add_argument("--limit", type=int, default=20)

    rp = sub.add_parser("receipt", help="print a clip's receipt")
    rp.add_argument("id", type=int)

    rr = sub.add_parser("render", help="re-render a clip's reel (e.g. after editing reel settings)")
    rr.add_argument("id", type=int)

    for name in ("approve", "reject"):
        sp = sub.add_parser(name, help=f"mark a clip {name}d")
        sp.add_argument("id", type=int)

    a = p.parse_args(argv)
    settings = load_settings(a.settings)
    channels = load_channels(a.channels)
    data_dir = (ROOT / settings["data_dir"]).resolve()
    q = lambda: ClipQueue(data_dir / "queue.db")  # noqa: E731

    if a.cmd == "demo":
        ch = next((c for c in channels if c["platform"] == "simulated"),
                  {"name": "demo_streamer", "platform": "simulated"})
        ch = {**ch, "sim": {**ch.get("sim", {}), "seed": a.seed}}
        queue = run_demo(settings, ch, a.minutes, not a.no_render, data_dir)
        print("\nRanked queue:")
        _print_queue(queue, None, 20)
    elif a.cmd == "run":
        try:
            asyncio.run(run_live(settings, channels, data_dir))
        except KeyboardInterrupt:
            print("stopped")
    elif a.cmd == "watch":
        from .engine import watch_chat
        try:
            asyncio.run(watch_chat(settings, {"name": a.channel, "platform": a.platform}))
        except KeyboardInterrupt:
            pass
    elif a.cmd == "queue":
        _print_queue(q(), a.status, a.limit)
    elif a.cmd == "receipt":
        c = q().get(a.id)
        print(json.dumps(c.receipt, indent=2) if c else "not found")
    elif a.cmd == "render":
        queue = q()
        c = queue.get(a.id)
        ch = next((x for x in channels if x["name"] == c.channel), {"name": c.channel})
        craft_clip(queue, a.id, channel_settings(settings, ch), ch.get("facecam"))
        print(queue.get(a.id).reel_path)
    elif a.cmd in ("approve", "reject"):
        q().update(a.id, status=a.cmd + "d")
        print(f"clip {a.id} {a.cmd}d")


if __name__ == "__main__":
    main()
