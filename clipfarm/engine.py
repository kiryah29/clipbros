"""Pipeline orchestration: chat -> detector -> recorder -> queue -> craft."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .chat import make_chat_source
from .chat.simulated import DEMO_LINES, SimulatedChat
from .config import channel_settings
from .craft import render_reel
from .craft.captions import fake_words, transcribe
from .detector import SpikeDetector, score_spike
from .models import Spike
from .queue import ClipQueue
from .recorder import DemoRecorder, LiveRecorder
from .scheduler import in_window


def clip_window(spike: Spike, clip_cfg: dict) -> tuple[float, float]:
    start = spike.peak_ts - clip_cfg["pre_roll_s"]
    end = max(spike.end_ts, spike.peak_ts) + clip_cfg["post_roll_s"]
    if end - start > clip_cfg["max_length_s"]:
        start = end - clip_cfg["max_length_s"]
    return start, end


def _paths(data_dir: Path, channel: str, clip_id: int) -> tuple[Path, Path, Path]:
    base = data_dir / "clips" / channel
    return base / f"{clip_id:05d}_raw.mp4", base / f"{clip_id:05d}_reel.mp4", \
        base / f"{clip_id:05d}_receipt.json"


def craft_clip(queue: ClipQueue, clip_id: int, cfg: dict, facecam: dict | None,
               words=None) -> None:
    clip = queue.get(clip_id)
    raw = Path(clip.raw_path)
    reel = raw.with_name(raw.name.replace("_raw", "_reel"))
    try:
        if words is None:
            words = transcribe(raw, cfg["reel"].get("transcriber", "none"))
        render_reel(raw, reel, cfg["reel"], facecam, words)
        queue.update(clip_id, reel_path=str(reel), status="crafted")
    except Exception as e:
        err = getattr(e, "stderr", b"") or b""
        queue.update(clip_id, status="failed")
        print(f"  ! craft failed for clip {clip_id}: {e} {err.decode(errors='ignore')[-400:]}")


# --------------------------------------------------------------------------- demo
def run_demo(settings: dict, channel: dict, minutes: float, render: bool,
             data_dir: Path) -> ClipQueue:
    cfg = channel_settings(settings, channel)
    queue = ClipQueue(data_dir / "queue.db")
    chat = SimulatedChat(channel["name"], minutes=minutes, **channel.get("sim", {}))
    det = SpikeDetector(channel["name"], cfg["detector"], cfg["ranking"]["hype_tokens"])
    rec = DemoRecorder(channel.get("facecam", {"x": 0.7, "y": 0.7, "w": 0.28, "h": 0.28}))

    print(f"Simulating {minutes:g} min of chat for '{channel['name']}' "
          f"({len(chat.bursts)} injected hype bursts)...")
    last = None
    for msg in chat.generate():
        det.feed(msg)
        last = msg.ts
    det.tick(last + 60)  # flush

    print(f"Detected {len(det.spikes)} spikes.\n")
    for i, sp in enumerate(det.spikes):
        start, end = clip_window(sp, cfg["clip"])
        score = score_spike(sp, cfg["ranking"])
        receipt = {"why": f"chat hit {sp.peak_rate} msg/s vs normal {sp.baseline_rate} "
                          f"({sp.ratio}x, z={sp.z_score})", **sp.to_dict()}
        cid = queue.add(channel["name"], channel["platform"], start, end, score, receipt)
        raw, _, rpath = _paths(data_dir, channel["name"], cid)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(json.dumps(receipt, indent=2))
        off = sp.peak_ts - chat.start_ts
        print(f"  clip {cid}: peak at +{off:6.1f}s  score={score:6.2f}  {receipt['why']}")
        if render:
            rec.cut(start, end, raw)
            queue.update(cid, raw_path=str(raw))
            words = fake_words(DEMO_LINES[i % len(DEMO_LINES)], start=cfg["reel"]["hook_seconds"])
            craft_clip(queue, cid, cfg, channel.get("facecam"), words=words)
    return queue


# --------------------------------------------------------------------------- live
class ChannelWorker:
    def __init__(self, settings: dict, channel: dict, queue: ClipQueue, data_dir: Path):
        self.ch = channel
        self.cfg = channel_settings(settings, channel)
        self.queue = queue
        self.data_dir = data_dir
        self.source = make_chat_source(channel)
        self.recorder: LiveRecorder | DemoRecorder | None = None
        self.pending: list[asyncio.Task] = []

    def _on_spike(self, sp: Spike) -> None:
        self.pending.append(asyncio.get_running_loop().create_task(self._handle(sp)))

    async def _handle(self, sp: Spike) -> None:
        start, end = clip_window(sp, self.cfg["clip"])
        await asyncio.sleep(max(0.0, end - time.time() + 3))  # let the buffer catch up
        score = score_spike(sp, self.cfg["ranking"])
        receipt = {"why": f"chat hit {sp.peak_rate} msg/s vs normal {sp.baseline_rate} "
                          f"({sp.ratio}x, z={sp.z_score})", **sp.to_dict()}
        cid = self.queue.add(self.ch["name"], self.ch["platform"], start, end, score, receipt)
        raw, _, rpath = _paths(self.data_dir, self.ch["name"], cid)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(json.dumps(receipt, indent=2))
        print(f"[{self.ch['name']}] spike -> clip {cid} (score {score}) {receipt['why']}")
        try:
            await asyncio.to_thread(self.recorder.cut, start, end, raw)
            self.queue.update(cid, raw_path=str(raw))
            await asyncio.to_thread(craft_clip, self.queue, cid, self.cfg, self.ch.get("facecam"))
        except Exception as e:
            self.queue.update(cid, status="failed")
            print(f"[{self.ch['name']}] clip {cid} failed: {e}")

    async def run(self) -> None:
        name = self.ch["name"]
        while not in_window(self.ch.get("schedule", [])):
            await asyncio.sleep(60)
        if self.ch["platform"] == "simulated":
            self.recorder = DemoRecorder(self.ch.get("facecam", {}))
        else:
            url = self.source.stream_url()
            self.recorder = LiveRecorder(name, url, self.data_dir, self.cfg["clip"]["quality"],
                                         self.cfg["clip"]["buffer_minutes"])
            self.recorder.start()
        det = SpikeDetector(name, self.cfg["detector"], self.cfg["ranking"]["hype_tokens"],
                            on_spike=self._on_spike)
        print(f"[{name}] monitoring {self.ch['platform']} chat")

        count = 0

        async def ticker():
            nonlocal count
            n = 0
            while True:
                await asyncio.sleep(1)
                det.tick(time.time())
                n += 1
                if n % 10 == 0:
                    base = f"{det.mean:.2f}" if det.mean is not None else "-"
                    print(f"[{name} {time.strftime('%H:%M:%S')}] {count / 10:.2f} msg/s | "
                          f"normal {base} | {'SPIKE OPEN' if det.open else 'watching'}")
                    count = 0
                if isinstance(self.recorder, LiveRecorder):
                    self.recorder.prune()
                if not in_window(self.ch.get("schedule", [])):
                    raise asyncio.CancelledError
        tick_task = asyncio.create_task(ticker())
        try:
            async for msg in self.source.messages():
                msg.ts = time.time()  # use arrival time on our clock (matches video buffer)
                det.feed(msg)
                count += 1
        finally:
            tick_task.cancel()
            if isinstance(self.recorder, LiveRecorder):
                self.recorder.stop()
            await asyncio.gather(*self.pending, return_exceptions=True)


async def run_live(settings: dict, channels: list[dict], data_dir: Path) -> None:
    queue = ClipQueue(data_dir / "queue.db")
    workers = [ChannelWorker(settings, c, queue, data_dir) for c in channels if c.get("enabled")]
    if not workers:
        print("No enabled channels in config/channels.yaml")
        return
    await asyncio.gather(*(w.run() for w in workers))


async def watch_chat(settings: dict, channel: dict) -> None:
    """Chat-only dry run: prints the pace every 10 s and any spike, records nothing."""
    cfg = channel_settings(settings, channel)
    src = make_chat_source(channel)

    def show(sp: Spike) -> None:
        print(f"\n*** SPIKE  peak {sp.peak_rate} msg/s vs normal {sp.baseline_rate} "
              f"({sp.ratio}x, z={sp.z_score})  score={score_spike(sp, cfg['ranking'])}")
        print(f"    top tokens: {sp.top_tokens[:5]}")
        print(f"    would clip {cfg['clip']['pre_roll_s']}s before the peak to "
              f"{cfg['clip']['post_roll_s']}s after it ends\n")

    det = SpikeDetector(channel["name"], cfg["detector"], cfg["ranking"]["hype_tokens"],
                        on_spike=show)
    count = 0
    t_start = time.time()
    print(f"Watching {channel['platform']} chat for '{channel['name']}' (Ctrl+C to stop). "
          f"Learning the normal pace for {cfg['detector']['warmup_s']} s first...")

    async def ticker():
        nonlocal count
        while True:
            await asyncio.sleep(10)
            det.tick(time.time())
            base = f"{det.mean:.2f}" if det.mean is not None else "-"
            state = "SPIKE OPEN" if det.open else ("warming up" if time.time() - t_start <
                                                   cfg["detector"]["warmup_s"] else "watching")
            print(f"[{time.strftime('%H:%M:%S')}] last 10 s: {count / 10:.2f} msg/s | "
                  f"normal: {base} msg/s | {state}")
            count = 0
    task = asyncio.create_task(ticker())
    try:
        async for msg in src.messages():
            msg.ts = time.time()
            det.feed(msg)
            count += 1
    finally:
        task.cancel()
