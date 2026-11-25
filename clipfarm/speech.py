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

Captions and hooks use a second, better pass: once a clip is cut, its 20-50 s of audio
is transcribed again with Parakeet TDT 0.6B (sherpa-onnx, real word timings, runs on
x64 and Windows ARM64). The small live model above stays for scoring only; on its own
it missed about a quarter of the words, invented lines and looped ("OH YEAH OH YEAH").
  model: python -m clipfarm setup-speech --clip
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

SHERPA_MODEL = "sherpa-onnx-whisper-base.en"   # tiny.en is faster but loops/mishears more
CLIP_MODEL = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"   # captions: one pass per clip
SHERPA_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{}.tar.bz2"
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
        self.max_tokens = str(cfg.get("max_tokens_per_second", 25))

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> list[Word]:
        s = self.rec.create_stream()
        try:   # default 6/s truncates fast talkers (15 still cut xQc off); drop_loops
            s.set_option("max_tokens_per_second", self.max_tokens)   # trims any loop
        except Exception:
            pass
        s.accept_waveform(sample_rate, [x / 32768 for x in samples(pcm)])
        self.rec.decode_stream(s)
        return voiced_words(s.result.text, pcm, sample_rate)


class Parakeet:
    """NVIDIA Parakeet TDT (English) through sherpa-onnx, for a whole clip at once.
    Unlike the live Whisper model it times every word, so captions land on the word."""
    name = "parakeet"
    window_s = 20.0     # the model is trained on short utterances; longer audio is cut up

    def __init__(self, cfg: dict):
        import sherpa_onnx
        d = Path(cfg.get("clip_model_dir") or ROOT / "models" / CLIP_MODEL)
        if not d.is_absolute():
            d = ROOT / d
        enc, dec, join = _pick(d, "encoder"), _pick(d, "decoder"), _pick(d, "joiner")
        tokens = next(d.glob("*tokens.txt"), None)
        if not (enc and dec and join and tokens):
            raise RuntimeError(f"no Parakeet model in {d} "
                               "(run: python -m clipfarm setup-speech --clip)")
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(enc), decoder=str(dec), joiner=str(join), tokens=str(tokens),
            model_type="nemo_transducer", num_threads=int(cfg.get("threads", 2)))

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> list[Word]:
        x = [v / 32768 for v in samples(pcm)]
        step = int(self.window_s * sample_rate)
        words: list[Word] = []
        for off in range(0, len(x), step):
            s = self.rec.create_stream()
            s.accept_waveform(sample_rate, x[off:off + step])
            self.rec.decode_stream(s)
            words += token_words(s.result.tokens, s.result.timestamps, off / sample_rate)
        return words

    def transcribe_file(self, media: Path) -> list[Word]:
        from .audio import decode_pcm
        return self.transcribe_pcm(decode_pcm(media))


def token_words(tokens, times, offset: float = 0.0, token_s: float = 0.08) -> list[Word]:
    """Sub-word tokens with start times -> words. A token starting with a space (or the
    sentencepiece mark) opens a new word; a word ends where the next one starts, at most
    0.6 s after its last token."""
    out: list[list] = []
    for tok, t in zip(tokens, times):
        t = float(t) + offset
        if tok.startswith((" ", "\u2581")) or not out:
            out.append([tok.lstrip(" \u2581"), t, t + token_s])
        else:
            out[-1][0] += tok
            out[-1][2] = t + token_s
    words = [Word(w, s, e) for w, s, e in out if w]
    for a, b in zip(words, words[1:]):
        a.end = max(a.end, min(b.start, a.end + 0.6))
    return words


def make_clip_transcriber(cfg: dict, quiet: bool = False):
    """The per-clip caption engine, or None if its model is missing."""
    try:
        return Parakeet(cfg)
    except Exception as e:
        if not quiet:
            print(f"  ! clip transcription is off, captions use the live transcript ({e})")
        return None


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


def voiced_words(text: str, pcm: bytes, sample_rate: int = SAMPLE_RATE,
                 frame_s: float = 0.02) -> list[Word]:
    """Words placed on the loud (voiced) parts of the chunk, longer words taking longer,
    for engines without word timings. Spreading evenly over the whole chunk put words
    seconds away from when they were said whenever the chunk had a pause."""
    words = text.split()
    dur = len(pcm) / 2 / sample_rate
    if not words:
        return []
    x = samples(pcm)
    n = int(sample_rate * frame_s)
    rms = []
    for i in range(0, len(x) - n + 1, n):     # first difference = cheap high-pass (less rumble)
        f = x[i:i + n]
        rms.append((sum((b - a) ** 2 for a, b in zip(f, f[1:])) / n) ** 0.5)
    if len(rms) < 5:
        return spread_words(text, 0.0, dur)
    srt = sorted(rms)
    lo, hi = srt[int(0.30 * (len(srt) - 1))], srt[int(0.95 * (len(srt) - 1))]
    voiced = [i for i, r in enumerate(rms) if r > lo + 0.3 * (hi - lo)]
    if len(voiced) < 3:
        return spread_words(text, 0.0, dur)
    weight = [len(_norm(w)) + 2 for w in words]
    total, acc, t = sum(weight), 0, [voiced[0] * frame_s]
    for wt in weight:
        acc += wt
        t.append(voiced[round(acc / total * (len(voiced) - 1))] * frame_s)
    return [Word(w, t[i], max(t[i + 1], t[i] + 0.08)) for i, w in enumerate(words)]


def _norm(w: str) -> str:
    return "".join(TOKEN_RE.findall(w.lower()))


def drop_loops(words: list[Word], max_repeat: int = 3, max_n: int = 20) -> list[Word]:
    """Cut Whisper hallucination loops: a short phrase ("no no no", "let's go let's go")
    is kept up to `max_repeat` times back to back, a longer one (4+ words, e.g. a
    whole sentence repeated 7x) only once. Sound tags like "(wind blowing)" go too."""
    words = [w for w in words if not w.text.startswith(("(", "["))
             and not w.text.endswith((")", "]"))]
    toks = [_norm(w.text) for w in words]
    out, i = [], 0
    while i < len(words):
        for n in range(1, max_n + 1):
            reps = 1
            while i + (reps + 1) * n <= len(toks) and                     toks[i + reps * n:i + (reps + 1) * n] == toks[i:i + n]:
                reps += 1
            keep = max_repeat if n <= 3 else 1
            if reps > keep:
                out += words[i:i + keep * n]
                i += reps * n
                break
        else:
            out.append(words[i])
            i += 1
    return out


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


def download_sherpa_model(dest: Path | None = None, model: str = SHERPA_MODEL) -> Path:
    """Fetch a sherpa-onnx model from its GitHub release (int8 weights + tokens only)."""
    dest = Path(dest or ROOT / "models")
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / model).exists():
        return dest / model
    tmp = dest / f"{model}.tar.bz2"
    print(f"Downloading {SHERPA_URL.format(model)}...")
    urllib.request.urlretrieve(SHERPA_URL.format(model), tmp)
    with tarfile.open(tmp) as t:
        keep = [m for m in t.getmembers() if m.isfile() and
                (m.name.endswith(".int8.onnx") or m.name.endswith("tokens.txt"))
                and ".." not in m.name and not m.name.startswith("/")]
        t.extractall(dest, members=keep)
    tmp.unlink(missing_ok=True)
    return dest / model


# ----------------------------------------------------------------------- scoring
def phrase_list(keywords) -> list[str]:
    return [" ".join(TOKEN_RE.findall(p.lower())) for p in keywords or []]


def find_phrases(toks: list[str], phrases: list[str], repeat_n: int = 3) -> list[str]:
    """Excited phrases in a run of lowercase tokens, plus repeated words ("no x3")."""
    text = " " + " ".join(toks) + " "
    found = [p for p in phrases if p and text.count(f" {p} ")
             for _ in range(text.count(f" {p} "))]
    run = 1
    for a, b in zip(toks, toks[1:]):           # "no no no", "go go go"
        run = run + 1 if a == b else 1
        if run == repeat_n:
            found.append(f"{a} x{repeat_n}")
    return found


class SpeechTracker:
    def __init__(self, cfg: dict, trigger_ratio: float = 3.0):
        self.cfg = cfg
        self.trigger_ratio = trigger_ratio
        self.phrases = phrase_list(cfg.get("keywords", []))
        self.phrase_target = float(cfg.get("phrase_target", 3))
        self.repeat_n = int(cfg.get("repeat_words", 3))
        self.max_wps = float(cfg.get("max_words_per_s", 6))
        # pace baseline: one value per transcribed chunk (words/s)
        self.pace = BaselineTrigger(halflife_s=float(cfg.get("baseline_halflife_s", 300)),
                                    warmup_s=0, min_ratio=1e9,
                                    floor=float(cfg.get("min_pace", 0.5)),
                                    history_s=int(cfg.get("keep_s", 900)))
        self.words: deque[Word] = deque(maxlen=20000)
        self.processed_until: float | None = None

    def add_chunk(self, start_ts: float, duration: float, words: list[Word]) -> None:
        """Words with times relative to the chunk start."""
        words = drop_loops(words, max(self.repeat_n, 3))
        for w in words:
            self.words.append(Word(w.text, start_ts + w.start, start_ts + w.end))
        # nobody says more than ~6 words/s; anything above is a transcription glitch
        self.pace.feed(start_ts, min(len(words) / max(duration, 0.5), self.max_wps))
        self.processed_until = start_ts + duration

    def words_between(self, start: float, end: float) -> list[Word]:
        """Words inside the window, re-timed to start at 0 (for captions)."""
        return [Word(w.text, w.start - start, w.end - start) for w in self.words
                if w.start >= start and w.end <= end]

    def phrase_hits(self, start: float, end: float) -> tuple[int, list[str]]:
        toks = [t for w in self.words if start <= w.start <= end
                for t in TOKEN_RE.findall(w.text.lower())]
        found = find_phrases(toks, self.phrases, self.repeat_n)
        return len(found), found

    def score(self, start: float, end: float) -> tuple[float, dict]:
        pace_ratio = self.pace.peak_ratio(start - 4, end)   # chunk ts = chunk start
        hits, found = self.phrase_hits(start, end)
        val = 0.5 * strength(pace_ratio, self.trigger_ratio) + \
            0.5 * min(1.0, hits / max(self.phrase_target, 1))
        return round(val, 3), {"pace_ratio": pace_ratio, "phrases": found[:8]}
