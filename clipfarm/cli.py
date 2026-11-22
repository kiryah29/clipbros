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

    r = sub.add_parser("run", help="monitor enabled live channels (or one: run twitch <name>)")
    r.add_argument("platform", nargs="?", choices=["twitch", "youtube", "kick"])
    r.add_argument("channel", nargs="?")
    r.add_argument("--game", default=None, help="game profile for a one-off channel, e.g. gta6")

    ss = sub.add_parser("setup-speech", help="download the speech-to-text model and test it")
    ss.add_argument("--model-dir", default=None)
    ss.add_argument("--model", choices=["base.en", "tiny.en"], default="base.en",
                    help="base.en is more accurate; tiny.en is faster on slow PCs")

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

    tp = sub.add_parser("tiktok-app", help="web app: connect TikTok, review a clip, post it (Content Posting API)")
    tp.add_argument("--host", default="127.0.0.1")
    tp.add_argument("--port", type=int, default=8000)

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
        if a.channel:   # one-off: this channel only, whatever channels.yaml says
            found = next((c for c in channels if c["name"].lower() == a.channel.lower()
                          and c["platform"] == a.platform), {})
            channels = [{**found, "name": a.channel, "platform": a.platform, "enabled": True,
                         "schedule": [], **({"game": a.game} if a.game else {})}]
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
    elif a.cmd == "setup-speech":
        from .speech import download_sherpa_model, make_transcriber
        try:
            import faster_whisper  # noqa: F401
            print("faster-whisper is installed; it is used first (engine: auto).")
        except ImportError:
            print("faster-whisper is not installed (it has no Windows ARM64 build); "
                  "using sherpa-onnx.")
        path = download_sherpa_model(Path(a.model_dir) if a.model_dir else None,
                                     f"sherpa-onnx-whisper-{a.model}")
        print(f"sherpa-onnx model: {path}")
        if a.model_dir or a.model != "base.en":
            print(f"Set speech.sherpa_model_dir: {path} in config/settings.yaml")
        t = make_transcriber({**settings["speech"], "sherpa_model_dir": str(path)})
        if t is None:
            sys.exit("Speech-to-text still unavailable: pip install sherpa-onnx")
        from .audio import SAMPLE_RATE
        t.transcribe_pcm(b"\x00\x00" * SAMPLE_RATE)   # loads + runs the model once
        print(f"Speech-to-text ready: {t.name}")
    elif a.cmd == "tiktok-app":
        from aiohttp import web
        from .tiktok.api import Config as TTConfig, TikTokError
        from .tiktok.webapp import build_app
        try:
            cfg = TTConfig.from_env()
        except TikTokError as ex:
            sys.exit(f"TikTok app not configured: {ex} (see README, 'Post to TikTok')")
        web.run_app(build_app(cfg, data_dir / "queue.db", data_dir / "tiktok_token.json"),
                    host=a.host, port=a.port)
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
