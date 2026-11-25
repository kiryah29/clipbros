from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .config import ROOT, channel_settings, load_channels, load_settings
from .agent import candidate_json
from .engine import craft_clip, run_demo, run_live
from .queue import ClipQueue


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def ranked_candidates(clips, include_low: bool = False) -> list[dict]:
    """ClipAgent candidates of queued clips, most confident (then highest score) first."""
    rows = [(c, c.receipt["clip_agent"]) for c in clips if c.receipt.get("clip_agent")]
    rows = [(c, ca) for c, ca in rows if include_low or ca.get("recommended")]
    rows.sort(key=lambda r: (r[1]["confidence_score"], r[0].score), reverse=True)
    return [{"clip_id": c.id, "channel": c.channel, **candidate_json(ca)} for c, ca in rows]


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
    p.add_argument("--data-dir", default=None, help="clips + queue.db folder (overrides settings data_dir)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the full pipeline on simulated chat (no API keys)")
    d.add_argument("--minutes", type=float, default=20)
    d.add_argument("--no-render", action="store_true", help="detect + queue only, skip ffmpeg")
    d.add_argument("--seed", type=int, default=7)

    r = sub.add_parser("run", help="monitor enabled live channels (or one: run twitch <name>)")
    r.add_argument("platform", nargs="?", choices=["twitch", "youtube", "kick"])
    r.add_argument("channel", nargs="?")
    r.add_argument("--game", default=None, help="game profile for a one-off channel, e.g. gta6")
    r.add_argument("--minutes", type=float, default=None, help="stop by itself after this many minutes")

    sub.add_parser("control-password", help="make the CONTROL_PASSWORD_HASH line for .env")

    ss = sub.add_parser("setup-speech", help="download the speech-to-text model and test it")
    ss.add_argument("--model-dir", default=None)
    ss.add_argument("--model", choices=["base.en", "tiny.en"], default="base.en",
                    help="base.en is more accurate; tiny.en is faster on slow PCs")
    ss.add_argument("--clip", action="store_true",
                    help="also download the caption model (Parakeet, ~460 MB) and the voice "
                         "model (~40 MB)")

    sn = sub.add_parser("setup-nudity", help="download the nudity-check model and test it")
    sn.add_argument("clip", nargs="?", help="optional: scan this video and print the result")

    w = sub.add_parser("watch", help="chat-only test: show live chat pace and spikes, no recording")
    w.add_argument("platform", choices=["twitch", "youtube", "kick"])
    w.add_argument("channel")

    qp = sub.add_parser("queue", help="show the ranked clip queue")
    qp.add_argument("--status", default=None)
    qp.add_argument("--limit", type=int, default=20)

    cp = sub.add_parser("candidates", help="ClipAgent candidates as JSON, ranked by confidence")
    cp.add_argument("--all", action="store_true", help="include clips under the confidence floor")
    cp.add_argument("--limit", type=int, default=50)
    rp = sub.add_parser("receipt", help="print a clip's receipt")
    rp.add_argument("id", type=int)

    rr = sub.add_parser("render", help="re-render a clip's reel (e.g. after editing reel settings)")
    rr.add_argument("id", type=int)

    for name in ("approve", "reject"):
        sp = sub.add_parser(name, help=f"mark a clip {name}d")
        sp.add_argument("id", type=int)

    vo = sub.add_parser("voice", help="teach the app a streamer's voice (captions skip game voices)")
    vo.add_argument("channel")
    vo.add_argument("--clip", type=int, help="clip id in this queue (or use --file)")
    vo.add_argument("--file", help="any video/audio file, e.g. a control run's NNNNN_raw.mp4")
    vo.add_argument("--from", dest="start", type=float, help="seconds into the file where only "
                    "the streamer talks")
    vo.add_argument("--to", dest="end", type=float)
    vo.add_argument("--forget", action="store_true", help="delete this streamer's voiceprint")

    hl = sub.add_parser("hooklab", help="today's hooks next to the rule writer's candidates, per clip")
    hl.add_argument("--run", default=None, help="only this control run id")
    tp = sub.add_parser("tiktok-app", help="web app: connect TikTok, review a clip, post it (Content Posting API)")
    tp.add_argument("--host", default="127.0.0.1")
    tp.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    settings = load_settings(a.settings)
    channels = load_channels(a.channels)
    data_dir = Path(a.data_dir).resolve() if a.data_dir else (ROOT / settings["data_dir"]).resolve()
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
        from .streamers import FILE as STREAMERS_FILE, StreamerRegistry
        reg = StreamerRegistry(data_dir / STREAMERS_FILE)
        for c in channels:
            if c.get("enabled") and c.get("platform") != "simulated":
                reg.record(c["name"], c["platform"], c.get("game"), source="cli")
        live = run_live(settings, channels, data_dir)
        failures = []
        try:
            failures = asyncio.run(asyncio.wait_for(live, a.minutes * 60) if a.minutes else live)
        except KeyboardInterrupt:
            print("stopped")
        except TimeoutError:
            print("time limit reached, stopped")
        if failures:   # non-zero exit, so the control page marks the run "failed"
            sys.exit("recording failed: " + "; ".join(failures))
    elif a.cmd == "control-password":
        import getpass
        from .control import hash_password
        pw = getpass.getpass("Shared password: ")
        if len(pw) < 10 or pw != getpass.getpass("Again: "):
            sys.exit("Passwords differ or are shorter than 10 characters.")
        print(f"CONTROL_PASSWORD_HASH={hash_password(pw)}")
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
        if a.clip:
            from .speech import CLIP_MODEL, make_clip_transcriber
            path = download_sherpa_model(Path(a.model_dir) if a.model_dir else None, CLIP_MODEL)
            print(f"caption model: {path}")
            if a.model_dir:
                print(f"Set speech.clip_model_dir: {path} in config/settings.yaml")
            c = make_clip_transcriber({**settings["speech"], "clip_model_dir": str(path)})
            if c is None:
                sys.exit("Caption model unavailable: pip install sherpa-onnx")
            c.transcribe_pcm(b"\x00\x00" * SAMPLE_RATE)
            print("Caption transcription ready: parakeet (reel.transcriber: clip)")
            from .voice import VoiceID, download_voice_model
            vpath = download_voice_model(Path(a.model_dir) if a.model_dir else None)
            VoiceID({**settings.get("voice", {}), "model": str(vpath)})
            print(f"voice model: {vpath} (teach a streamer's voice: clipfarm voice <channel> ...)")
    elif a.cmd == "voice":
        from .audio import decode_pcm
        from .voice import VoicePrint, enrol
        vp = VoicePrint.load(settings, a.channel)
        if a.forget:
            vp.path.unlink(missing_ok=True)
            print(f"forgot {a.channel}'s voice")
        elif a.start is None or a.end is None or not (a.clip or a.file):
            print(f"{a.channel}: " + (f"voiceprint from {vp.seconds:.0f} s in {len(vp.takes)} take(s)"
                                      if vp.embedding else "no voiceprint yet"))
            print("add one: clipfarm voice <channel> --file <clip.mp4> --from <s> --to <s>")
        else:
            src = Path(a.file) if a.file else Path(q().get(a.clip).raw_path)
            vp = enrol(settings, a.channel, decode_pcm(src), a.start, a.end,
                       note=f"{src.name} {a.start:g}-{a.end:g}s")
            print(f"{a.channel}: voiceprint from {vp.seconds:.0f} s in {len(vp.takes)} take(s) "
                  f"-> {vp.path}")
    elif a.cmd == "setup-nudity":
        from . import nudity
        cfg = nudity.settings(settings.get("nudity"))
        det = nudity.load_model({**cfg, "auto_download": True})
        if det is None:
            sys.exit("Nudity check unavailable (see the message above); pip install onnxruntime")
        print(f"Nudity check ready: {nudity.model_path(cfg)} ({det.engine})")
        if a.clip:
            print(json.dumps(nudity.check_clip(Path(a.clip), cfg, det), indent=2))
    elif a.cmd == "hooklab":
        from .hooklab import ReadOnlyRuns, print_lab
        print_lab(ReadOnlyRuns(data_dir / "control"), a.run)
    elif a.cmd == "tiktok-app":
        from aiohttp import web
        from .tiktok.api import Config as TTConfig, TikTokError
        from .tiktok.webapp import build_app
        try:
            cfg = TTConfig.from_env()
        except TikTokError as ex:
            sys.exit(f"TikTok app not configured: {ex} (see README, 'Post to TikTok')")
        from .recorder import _tie_children_to_us
        _tie_children_to_us()   # control-page runs die with the app
        from .instagram.api import Config as IGConfig
        web.run_app(build_app(cfg, data_dir / "queue.db", data_dir / "tiktok_token.json",
                              ig_cfg=IGConfig.from_env(),
                              password_hash=os.getenv("CONTROL_PASSWORD_HASH") or None,
                              allow_posting=os.getenv("CONTROL_ALLOW_POSTING") == "1",
                              keep_days=float(os.getenv("CONTROL_KEEP_DAYS") or 3)),
                    host=a.host, port=a.port)
    elif a.cmd == "queue":
        _print_queue(q(), a.status, a.limit)
    elif a.cmd == "candidates":
        print(json.dumps(ranked_candidates(q().ranked(None, 10000), a.all)[:a.limit],
                         indent=2))
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
