"""Pipeline orchestration: chat -> detector -> recorder -> queue -> craft."""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

from .chat import make_chat_source
from .chat.simulated import DEMO_LINES, SimulatedChat
from .config import channel_settings
from .craft import render_reel
from .audio import SAMPLE_RATE, AudioTracker, decode_pcm
from .craft.captions import Word, fake_words, transcribe
from .detector import SpikeDetector, chat_values, score_spike
from .models import Spike
from .moments import classify, clip_window_for, pick_hook
from .queue import ClipQueue
from .recorder import DemoRecorder, LiveRecorder
from .scheduler import in_window
from .signals import score_signals, strength
from .speech import SpeechTracker, make_transcriber


def _fit(start: float, end: float, c: dict) -> tuple[float, float]:
    if end - start > c["max_length_s"]:
        start = end - c["max_length_s"]
    if end - start < c.get("min_length_s", 0):
        start = end - c["min_length_s"]
    return start, end


def clip_window(spike: Spike, clip_cfg: dict, moments: list[dict] | None = None
                ) -> tuple[float, float]:
    c = clip_window_for(spike.moment, moments or [], clip_cfg)
    end = max(spike.end_ts, spike.peak_ts) + c["post_roll_s"]
    return _fit(spike.peak_ts - c["pre_roll_s"], end, c)


def audio_window(ev: dict, audio_cfg: dict, clip_cfg: dict) -> tuple[float, float]:
    return _fit(ev["peak_ts"] - float(audio_cfg.get("pre_roll_s", 15)),
                max(ev["end_ts"], ev["peak_ts"]) + float(audio_cfg.get("post_roll_s", 6)),
                clip_cfg)


def make_detector(name: str, cfg: dict, on_spike=None) -> SpikeDetector:
    moments = cfg.get("moments") or []
    return SpikeDetector(name, cfg["detector"], cfg["ranking"]["hype_tokens"], on_spike=on_spike,
                         classifier=(lambda msgs: classify(msgs, moments)) if moments else None)


class MomentBook:
    """Turns triggers (a chat spike, a loudness spike) into clip windows. A trigger that
    overlaps a clip still waiting to be cut is merged into it, so a scream plus the chat
    reaction a few seconds later make ONE clip, not two."""

    def __init__(self, clip_cfg: dict):
        self.clip = clip_cfg
        self.items: list[dict] = []

    def add(self, start: float, end: float, trigger: str, spike: Spike | None = None,
            audio: dict | None = None) -> tuple[dict | None, bool]:
        for m in self.items:
            if start <= m["end"] and end >= m["start"]:
                if m["done"]:                      # already cut: keep only the new part
                    start = m["end"]
                    if end - start < max(self.clip.get("min_length_s", 0), 10):
                        return None, False
                    continue
                m["start"], m["end"] = _fit(min(m["start"], start), max(m["end"], end), self.clip)
                m["triggers"].append(trigger)
                m["spike"] = m["spike"] or spike
                m["audio"] = m["audio"] or audio
                return m, False
        m = dict(start=start, end=end, triggers=[trigger], spike=spike, audio=audio, done=False)
        self.items.append(m)
        self.items = self.items[-50:]
        return m, True


def score_moment(m: dict, cfg: dict, chat: SpikeDetector | None, audio: AudioTracker | None,
                 speech: SpeechTracker | None) -> tuple[float, dict]:
    """All signals for a clip window -> (score 0-100, receipt)."""
    start, end, sp = m["start"], m["end"], m["spike"]
    trig = float(cfg["signals"].get("trigger_ratio", 3.0))
    chat_ratio = chat.ratio_between(start, end) if chat else 1.0
    values = chat_values(sp, chat_ratio, trig)
    why = []
    if sp:
        why.append(f"chat hit {sp.peak_rate} msg/s vs normal {sp.baseline_rate} "
                   f"({sp.ratio}x, z={sp.z_score})")
    elif chat_ratio > 1.2:
        why.append(f"chat {chat_ratio}x normal")
    extra: dict = {"chat_ratio": chat_ratio}
    if audio:
        a_ratio = audio.ratio_between(start, end)
        values["audio"] = strength(a_ratio, trig)
        extra["audio_ratio"] = a_ratio
        if a_ratio >= 1.5:
            line = f"audio {a_ratio}x normal loudness"
            why.insert(0, line) if m["triggers"][0] == "audio" else why.append(line)
    if speech:
        values["speech"], details = speech.score(start, end)
        extra["speech"] = details
        if details["phrases"]:
            why.append("said: " + ", ".join(f'"{p}"' for p in details["phrases"][:3]))
        if details["pace_ratio"] >= 1.5:
            why.append(f"talking {details['pace_ratio']}x normal pace")
    score, points = score_signals(values, cfg["signals"]["weights"])
    r = {"why": "; ".join(why) or "triggered", "triggers": m["triggers"], "score_points": points,
         "signals": values, **extra}
    if sp:
        r.update(sp.to_dict())
    if sp and sp.moment:
        r["why"] += f"; looks like: {sp.moment['label']} ({sp.moment['hits']} matching msgs)"
        r["hook"] = pick_hook(sp.moment, cfg["moments"], cfg["reel"].get("hook_text", ""),
                              cfg["streamer"], cfg.get("game_name", ""), seed=int(sp.peak_ts))
        r["safety_review"] = sp.moment.get("safety") == "review"
    if speech:
        r["transcript"] = [[w.text, round(w.start, 2), round(w.end, 2)]
                           for w in speech.words_between(start, end)]
    return score, r


def unconfirmed(receipt: dict, audio_cfg: dict) -> str | None:
    """Why an audio-only clip is dropped, or None to keep it. Loud alone is often just a
    car crash or game noise; keep it only if chat reacts or the streamer says something."""
    if not audio_cfg.get("confirm", True) or receipt["triggers"] != ["audio"]:
        return None
    need = float(audio_cfg.get("confirm_chat_ratio", 1.5))
    if receipt.get("chat_ratio", 1.0) >= need or receipt.get("speech", {}).get("phrases"):
        return None
    return f"audio only (chat {receipt.get('chat_ratio', 1.0)}x < {need}x, no excited phrase)"


def build_receipt(sp: Spike, cfg: dict) -> dict:
    """Chat-only receipt (used by `watch`)."""
    start, end = clip_window(sp, cfg["clip"], cfg["moments"])
    return score_moment(dict(start=start, end=end, triggers=["chat"], spike=sp, audio=None),
                        cfg, None, None, None)[1]


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
        if words is None and clip.receipt.get("transcript") and \
                cfg["reel"].get("transcriber", "auto") == "auto":
            words = [Word(*w) for w in clip.receipt["transcript"]]   # live stream transcript
        if words is None:
            words = transcribe(raw, cfg["reel"].get("transcriber", "none"))
        render_reel(raw, reel, cfg["reel"], facecam, words, hook_text=clip.receipt.get("hook"))
        queue.update(clip_id, reel_path=str(reel), status="crafted")
    except Exception as e:
        err = getattr(e, "stderr", b"") or b""
        queue.update(clip_id, status="failed")
        print(f"  ! craft failed for clip {clip_id}: {e} {err.decode(errors='ignore')[-400:]}")


# --------------------------------------------------------------------------- demo
DEMO_TALK = "okay so we go over here and grab the car then head to the shop".split()


def _demo_signals(chat: SimulatedChat, audio: AudioTracker | None, speech: SpeechTracker | None,
                  seed: int, chunk_s: float = 4.0) -> list[float]:
    """Fake stream audio + transcript that line up with the simulated chat: the streamer
    yells a few seconds before each chat burst, plus a few screams chat doesn't react to.
    Returns the offsets of those audio-only screams."""
    rng = random.Random(seed)
    dur = int(chat.duration)
    loud = [0.05 * rng.uniform(0.8, 1.25) for _ in range(dur)]
    yells = [(int(b) - 5, rng.uniform(3.5, 8)) for b, _, _ in chat.bursts]
    solo, t = [], 260.0
    while t < dur - 60:                              # screams chat doesn't react to
        if all(abs(t - b) > 40 for b, _, _ in chat.bursts):
            solo.append(t)
            yells.append((int(t), rng.uniform(4, 7)))
        t += rng.uniform(300, 500)
    for start, mult in yells:
        for i in range(start, min(start + 4, dur)):
            loud[i] = 0.05 * mult
    if audio:
        audio.feed_levels(chat.start_ts, loud)
    if speech:
        hot = {s for s, _ in yells}
        for c in range(0, dur, int(chunk_s)):
            if any(c <= s < c + chunk_s or c <= s + 2 < c + chunk_s for s in hot):
                text = rng.choice(DEMO_LINES) + " " + rng.choice(["no way", "oh my god",
                                                                  "let's go", "no no no"])
            else:
                text = " ".join(rng.sample(DEMO_TALK, rng.randint(5, 9)))
            speech.add_chunk(chat.start_ts + c, chunk_s, fake_words(text, 0, chunk_s /
                                                                    max(len(text.split()), 1)))
    return solo


def run_demo(settings: dict, channel: dict, minutes: float, render: bool,
             data_dir: Path) -> ClipQueue:
    cfg = channel_settings(settings, channel)
    queue = ClipQueue(data_dir / "queue.db")
    chat = SimulatedChat(channel["name"], minutes=minutes, **channel.get("sim", {}))
    det = make_detector(channel["name"], cfg)
    audio = AudioTracker(cfg["audio"]) if cfg["audio"].get("enabled", True) else None
    speech = SpeechTracker(cfg["speech"], cfg["signals"].get("trigger_ratio", 3.0))
    rec = DemoRecorder(channel.get("facecam", {"x": 0.7, "y": 0.7, "w": 0.28, "h": 0.28}))

    solo = _demo_signals(chat, audio, speech, channel.get("sim", {}).get("seed", 7))
    print(f"Simulating {minutes:g} min of chat + stream audio for '{channel['name']}' "
          f"({len(chat.bursts)} hype bursts, {len(solo)} screams chat ignores)...")
    last = None
    for msg in chat.generate():
        det.feed(msg)
        last = msg.ts
    det.tick(last + 60)  # flush

    book = MomentBook(cfg["clip"])
    triggers = [(sp.peak_ts, "chat", sp) for sp in det.spikes] + \
        [(ev["peak_ts"], "audio", ev) for ev in (audio.trigger.events if audio else [])]
    for _, kind, x in sorted(triggers, key=lambda t: t[0]):
        if kind == "chat":
            book.add(*clip_window(x, cfg["clip"], cfg["moments"]), "chat", spike=x)
        else:
            book.add(*audio_window(x, cfg["audio"], cfg["clip"]), "audio", audio=x)

    print(f"Detected {len(det.spikes)} chat spikes, "
          f"{len(audio.trigger.events) if audio else 0} loudness spikes "
          f"-> {len(book.items)} clips.\n")
    for i, m in enumerate(book.items):
        score, receipt = score_moment(m, cfg, det, audio, speech)
        if why_not := unconfirmed(receipt, cfg["audio"]):
            print(f"  skipped: {why_not}")
            continue
        cid = queue.add(channel["name"], channel["platform"], m["start"], m["end"], score, receipt)
        raw, _, rpath = _paths(data_dir, channel["name"], cid)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(json.dumps(receipt, indent=2))
        off = m["start"] - chat.start_ts
        print(f"  clip {cid}: +{off:6.1f}s  score={score:5.1f}  [{'+'.join(m['triggers'])}] "
              f"{receipt['why']}")
        if render:
            rec.cut(m["start"], m["end"], raw)
            queue.update(cid, raw_path=str(raw))
            words = fake_words(DEMO_LINES[i % len(DEMO_LINES)], start=0.6)
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
        self.book = MomentBook(self.cfg["clip"])
        self.det: SpikeDetector | None = None
        self.audio: AudioTracker | None = None
        self.speech: SpeechTracker | None = None
        self.transcriber = None

    def _trigger(self, kind: str, start: float, end: float, **what) -> None:
        m, new = self.book.add(start, end, kind, **what)
        if m is None:
            return
        if new:
            print(f"[{self.ch['name']}] {kind} trigger, clipping "
                  f"{time.strftime('%H:%M:%S', time.localtime(m['start']))}-"
                  f"{time.strftime('%H:%M:%S', time.localtime(m['end']))}")
            self.pending.append(asyncio.get_running_loop().create_task(self._handle(m)))

    def _on_spike(self, sp: Spike) -> None:
        self._trigger("chat", *clip_window(sp, self.cfg["clip"], self.cfg["moments"]), spike=sp)

    def _on_audio(self, ev: dict) -> None:
        self._trigger("audio", *audio_window(ev, self.cfg["audio"], self.cfg["clip"]), audio=ev)

    async def _handle(self, m: dict) -> None:
        name = self.ch["name"]
        while (wait := m["end"] - time.time() + 3) > 0:   # end can move if triggers merge
            await asyncio.sleep(wait)
        m["done"] = True
        deadline = time.time() + float(self.cfg["speech"].get("max_wait_s", 30))
        while time.time() < deadline and (   # let audio analysis / transcript catch up
                (self.audio and (self.audio.last_ts or 0) < m["end"] - 2 and
                 isinstance(self.recorder, LiveRecorder) and self.recorder.running) or
                (self.transcriber and (self.speech.processed_until or 0) < m["end"] - 1)):
            await asyncio.sleep(1)
        score, receipt = score_moment(m, self.cfg, self.det, self.audio,
                                      self.speech if self.transcriber else None)
        if why_not := unconfirmed(receipt, self.cfg["audio"]):
            print(f"[{name}] skipped (score {score}): {why_not}")
            return
        cid = self.queue.add(name, self.ch["platform"], m["start"], m["end"], score, receipt)
        raw, _, rpath = _paths(self.data_dir, name, cid)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(json.dumps(receipt, indent=2))
        print(f"[{name}] clip {cid} (score {score}) {receipt['why']}")
        try:
            await asyncio.to_thread(self.recorder.cut, m["start"], m["end"], raw)
            self.queue.update(cid, raw_path=str(raw))
            await asyncio.to_thread(craft_clip, self.queue, cid, self.cfg, self.ch.get("facecam"))
        except Exception as e:
            self.queue.update(cid, status="failed")
            print(f"[{name}] clip {cid} failed: {e}")

    async def _analyse_audio(self) -> None:
        """Decode each finished buffer segment once: loudness -> audio tracker,
        PCM -> speech chunks for the transcriber."""
        seen: set[str] = set()
        buf, buf_start = b"", None
        chunk_bytes = int(float(self.cfg["speech"].get("chunk_s", 4)) * SAMPLE_RATE) * 2
        q: asyncio.Queue = asyncio.Queue()
        backlog = int(self.cfg["speech"].get("max_backlog", 6))
        worker = asyncio.create_task(self._transcribe(q)) if self.transcriber else None
        try:
            while True:
                await asyncio.sleep(1)
                for f, a, _b in self.recorder.segments():
                    if f.name in seen:
                        continue
                    seen.add(f.name)
                    pcm = await asyncio.to_thread(decode_pcm, f)
                    if self.audio:
                        self.audio.feed_pcm(a, pcm)
                    if not worker:
                        continue
                    if buf_start is None or abs(buf_start + len(buf) / 2 / SAMPLE_RATE - a) > 1:
                        buf, buf_start = b"", a          # gap in the stream: start over
                    buf += pcm
                    if len(buf) >= chunk_bytes:
                        if q.qsize() >= backlog:
                            q.get_nowait()
                            print(f"[{self.ch['name']}] speech-to-text is behind, "
                                  "skipping a chunk (try a smaller model)")
                        q.put_nowait((buf_start, buf))
                        buf_start += len(buf) / 2 / SAMPLE_RATE
                        buf = b""
                if len(seen) > 2000:
                    seen = {f.name for f, _, _ in self.recorder.segments()}
        finally:
            if worker:
                worker.cancel()

    async def _transcribe(self, q: asyncio.Queue) -> None:
        while True:
            start, pcm = await q.get()
            try:
                words = await asyncio.to_thread(self.transcriber.transcribe_pcm, pcm)
            except Exception as e:
                print(f"[{self.ch['name']}] transcription failed: {e}")
                words = []
            self.speech.add_chunk(start, len(pcm) / 2 / SAMPLE_RATE, words)

    async def run(self) -> None:
        while True:
            while not in_window(self.ch.get("schedule", [])):
                await asyncio.sleep(60)
            await self._run_window()
            if not self.ch.get("schedule"):
                return

    async def _run_window(self) -> None:
        """Record + watch until the chat ends, the schedule window closes or we're stopped.
        Always stops streamlink/ffmpeg on the way out."""
        name = self.ch["name"]
        live = self.ch["platform"] != "simulated"
        if live:
            url = self.source.stream_url()
            self.recorder = LiveRecorder(name, url, self.data_dir, self.cfg["clip"]["quality"],
                                         self.cfg["clip"]["buffer_minutes"])
            self.recorder.start()
        else:
            self.recorder = DemoRecorder(self.ch.get("facecam", {}))
        tasks: list[asyncio.Task] = []
        try:
            self.det = make_detector(name, self.cfg, on_spike=self._on_spike)
            if live and self.cfg["audio"].get("enabled", True):
                self.audio = AudioTracker(self.cfg["audio"], on_event=self._on_audio)
            self.speech = SpeechTracker(self.cfg["speech"],
                                        self.cfg["signals"].get("trigger_ratio", 3.0))
            if live:
                self.transcriber = await asyncio.to_thread(make_transcriber, self.cfg["speech"])
            print(f"[{name}] monitoring {self.ch['platform']} chat"
                  f"{' + stream audio' if self.audio else ''}"
                  f"{f' + speech ({self.transcriber.name})' if self.transcriber else ''}")
            tasks.append(asyncio.create_task(self._consume()))
            tasks.append(asyncio.create_task(self._ticker()))
            if live:
                tasks.append(asyncio.create_task(self._analyse_audio()))
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception():
                    print(f"[{name}] stopped: {t.exception()!r}")
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if isinstance(self.recorder, LiveRecorder):
                self.recorder.stop()
                print(f"[{name}] recording stopped")
            await asyncio.gather(*self.pending, return_exceptions=True)

    async def _consume(self) -> None:
        self.msg_count = 0
        async for msg in self.source.messages():
            msg.ts = time.time()  # use arrival time on our clock (matches video buffer)
            self.det.feed(msg)
            self.msg_count += 1

    async def _ticker(self) -> None:
        """Closes chat buckets during silence, prints status, prunes the buffer, and
        returns (ending the window) once the schedule window closes."""
        name, n = self.ch["name"], 0
        while True:
            await asyncio.sleep(1)
            self.det.tick(time.time())
            n += 1
            if n % 10 == 0:
                d = self.det
                base = f"{d.mean:.2f}" if d.mean is not None else "-"
                loud = (f" | loudness {self.audio.trigger.history[-1][1]:.3f} "
                        f"(normal {self.audio.baseline:.3f})"
                        if self.audio and self.audio.trigger.history else "")
                print(f"[{name} {time.strftime('%H:%M:%S')}] {self.msg_count / 10:.2f} msg/s | "
                      f"normal {base}{loud} | {'SPIKE OPEN' if d.open else 'watching'}")
                self.msg_count = 0
            if isinstance(self.recorder, LiveRecorder):
                self.recorder.prune()
                if not self.recorder.running:
                    print(f"[{name}] stream recorder exited (stream offline?)")
                    return
            if not in_window(self.ch.get("schedule", [])):
                print(f"[{name}] schedule window closed")
                return


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
              f"({sp.ratio}x, z={sp.z_score})  score={score_spike(sp, cfg['signals']['weights'])}")
        print(f"    top tokens: {sp.top_tokens[:5]}")
        c = clip_window_for(sp.moment, cfg["moments"], cfg["clip"])
        if sp.moment:
            r = build_receipt(sp, cfg)
            flag = "  [HOLD FOR SAFETY REVIEW]" if r["safety_review"] else ""
            print(f"    moment: {sp.moment['label']}  hook: \"{r['hook']}\"{flag}")
        print(f"    would clip {c['pre_roll_s']}s before the peak to "
              f"{c['post_roll_s']}s after it ends\n")

    det = make_detector(channel["name"], cfg, on_spike=show)
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
