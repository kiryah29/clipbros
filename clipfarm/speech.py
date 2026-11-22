"""Streamer speech signal (and captions from the same transcript).

The stream audio is transcribed in short chunks while the stream runs. Two things
are scored inside a clip window:
  pace      words per second vs the streamer's normal pace (rapid-fire talking)
  phrases   excited phrases ("no way", "oh my god", "let's go") and repeated words
            ("no no no no")
speech = 0.5 * strength(pace ratio) + 0.5 * min(1, phrase hits / phrase_target)

Speech-to-text engines (settings `speech.engine`):
  faster_whisper  pip install faster-whisper   exact word timings; x64 PCs only
  sherpa_onnx     pip install sherpa-onnx      runs on Windows ARM64 too;
                  model: python -m clipfarm setup-speech
  auto            faster_whisper, else sherpa_onnx, else off
  none            off (speech scores 0)
"""
from __future__ import annotations

import re
import tarfile
import urllib.request
from collections import deque
from pathlib import Path

from .audio import SAMPLE_RATE, samples
from .config import ROOT
from .craft.captions import Word
from .signals import BaselineTrigger, strength

SHERPA_MODEL = "sherpa-onnx-whisper-tiny.en"
SHERPA_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
              f"{SHERPA_MODEL}.tar.bz2")
TOKEN_RE = re.compile(r"[a-z0-9']+")


# ----------------------------------------------------------------------- engines
class FasterWhisper:
    name = "faster_whisper"

    def __init__(self, cfg: dict):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(cfg.get("whisper_model", "base.en"), compute_type="int8")
        self.language = cfg.get("language") or None

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> list[Word]:
        import numpy as np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        segs, _ = self.model.transcribe(audio, language=self.language, word_timestamps=True,
                                        vad_filter=True)
        return [Word(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]


class SherpaOnnx:
    name = "sherpa_onnx"

    def __init__(self, cfg: dict):
        import sherpa_onnx
        d = Path(cfg.get("sherpa_model_dir") or ROOT / "models" / SHERPA_MODEL)
        if not d.is_absolute():
            d = ROOT / d
        enc, dec = _pick(d, "encoder"), _pick(d, "decoder")
        tokens = next(d.glob("*tokens.txt"), None)
        if not (enc and dec and tokens):
            raise RuntimeError(f"no sherpa-onnx whisper model in {d} "
                               "(run: python -m clipfarm setup-speech)")
        self.rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(enc), decoder=str(dec), tokens=str(tokens),
            language=cfg.get("language", "en"), num_threads=int(cfg.get("threads", 2)))

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> list[Word]:
        s = self.rec.create_stream()
        try:   # streamers talk fast; the default 6 tokens/s cap truncates them
            s.set_option("max_tokens_per_second", "25")
        except Exception:
            pass
        s.accept_waveform(sample_rate, [x / 32768 for x in samples(pcm)])
        self.rec.decode_stream(s)
        dur = len(pcm) / 2 / sample_rate
        return spread_words(s.result.text, 0.0, dur)


def _pick(d: Path, part: str) -> Path | None:
    files = sorted(d.glob(f"*{part}*.onnx"))
    return next((f for f in files if "int8" in f.name), files[0] if files else None)


def spread_words(text: str, start: float, end: float) -> list[Word]:
    """Words evenly over [start, end] for engines without word timings."""
    words = text.split()
    if not words:
        return []
    step = (end - start) / len(words)
    return [Word(w, start + i * step, start + (i + 1) * step) for i, w in enumerate(words)]


def make_transcriber(cfg: dict, quiet: bool = False):
    """The configured engine, or None (speech off). Never raises for a missing engine."""
    engine = (cfg.get("engine") or "none").lower()
    order = {"auto": [FasterWhisper, SherpaOnnx], "faster_whisper": [FasterWhisper],
             "sherpa_onnx": [SherpaOnnx]}.get(engine, [])
    errors = []
    for cls in order:
        try:
            return cls(cfg)
        except Exception as e:  # ImportError, missing model...
            errors.append(f"{cls.name}: {e}")
    if order and not quiet:
        print("  ! speech-to-text is off, speech scores 0 (" + "; ".join(errors) + ")")
    return None


def download_sherpa_model(dest: Path | None = None) -> Path:
    dest = Path(dest or ROOT / "models")
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / SHERPA_MODEL).exists():
        return dest / SHERPA_MODEL
    tmp = dest / f"{SHERPA_MODEL}.tar.bz2"
    print(f"Downloading {SHERPA_URL} (~120 MB)...")
    urllib.request.urlretrieve(SHERPA_URL, tmp)
    with tarfile.open(tmp) as t:
        keep = [m for m in t.getmembers() if m.isfile() and
                (m.name.endswith(".int8.onnx") or m.name.endswith("tokens.txt"))
                and ".." not in m.name and not m.name.startswith("/")]
        t.extractall(dest, members=keep)
    tmp.unlink(missing_ok=True)
    return dest / SHERPA_MODEL


# ----------------------------------------------------------------------- scoring
class SpeechTracker:
    def __init__(self, cfg: dict, trigger_ratio: float = 3.0):
        self.cfg = cfg
        self.trigger_ratio = trigger_ratio
        self.phrases = [" ".join(TOKEN_RE.findall(p.lower())) for p in cfg.get("keywords", [])]
        self.phrase_target = float(cfg.get("phrase_target", 3))
        self.repeat_n = int(cfg.get("repeat_words", 3))
        # pace baseline: one value per transcribed chunk (words/s)
        self.pace = BaselineTrigger(halflife_s=float(cfg.get("baseline_halflife_s", 300)),
                                    warmup_s=0, min_ratio=1e9,
                                    floor=float(cfg.get("min_pace", 0.5)),
                                    history_s=int(cfg.get("keep_s", 900)))
        self.words: deque[Word] = deque(maxlen=20000)
        self.processed_until: float | None = None

    def add_chunk(self, start_ts: float, duration: float, words: list[Word]) -> None:
        """Words with times relative to the chunk start."""
        for w in words:
            self.words.append(Word(w.text, start_ts + w.start, start_ts + w.end))
        self.pace.feed(start_ts, len(words) / max(duration, 0.5))
        self.processed_until = start_ts + duration

    def words_between(self, start: float, end: float) -> list[Word]:
        """Words inside the window, re-timed to start at 0 (for captions)."""
        return [Word(w.text, w.start - start, w.end - start) for w in self.words
                if w.start >= start and w.end <= end]

    def phrase_hits(self, start: float, end: float) -> tuple[int, list[str]]:
        toks = [t for w in self.words if start <= w.start <= end
                for t in TOKEN_RE.findall(w.text.lower())]
        text = " " + " ".join(toks) + " "
        found = [p for p in self.phrases if p and text.count(f" {p} ")
                 for _ in range(text.count(f" {p} "))]
        run = 1
        for a, b in zip(toks, toks[1:]):           # "no no no", "go go go"
            run = run + 1 if a == b else 1
            if run == self.repeat_n:
                found.append(f"{a} x{self.repeat_n}")
        return len(found), found

    def score(self, start: float, end: float) -> tuple[float, dict]:
        pace_ratio = self.pace.peak_ratio(start - 4, end)   # chunk ts = chunk start
        hits, found = self.phrase_hits(start, end)
        val = 0.5 * strength(pace_ratio, self.trigger_ratio) + \
            0.5 * min(1.0, hits / max(self.phrase_target, 1))
        return round(val, 3), {"pace_ratio": pace_ratio, "phrases": found[:8]}
