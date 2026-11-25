"""The streamer's own voice, so captions and hook quotes skip game characters.

The stream is one mixed audio track: the streamer's mic, cutscene dialogue, police radio,
phone calls and Discord guests all land in the same transcript. A voiceprint (a speaker
embedding: NeMo TitaNet small through sherpa-onnx, ~40 MB) is kept per streamer in
data/voices/<channel>.json. Every caption sentence is compared with it:

  below voice.min_similarity        not the streamer: no caption
  below voice.quote_min_similarity  captioned, but never quoted on the hook card

Enrol a streamer once from a clip stretch where only they talk:
  python -m clipfarm voice <channel> --clip <id> --from 5 --to 17
Enrolling again (another clip) averages in, so the print gets better with more takes.

Measured on nickmercs (GTA 6, 25 Nov): enrolled from 12 s of him talking, his sentences
scored 0.16-0.60 (one at -0.02), police radio 0.03-0.14, a cutscene phone call -0.02-0.18.
The streamer's mic is mixed with game sound, so short or overlapping lines score lower:
the caption cut is set low on purpose (missing a streamer line is worse than keeping an
NPC line), the quote cut high. There is no automatic enrolment: on a 100 s test stretch
the most frequent voice was the police dispatcher, not the streamer.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

from .audio import SAMPLE_RATE, samples
from .config import ROOT
from .craft.captions import Word

VOICE_MODEL = "nemo_en_titanet_small.onnx"
VOICE_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/{}")    # sic: the release tag is spelled this way
SENTENCE_END = (".", "!", "?")


def model_path(cfg: dict) -> Path:
    p = Path(cfg.get("model") or ROOT / "models" / VOICE_MODEL)
    return p if p.is_absolute() else ROOT / p


def download_voice_model(dest: Path | None = None) -> Path:
    dest = Path(dest or ROOT / "models")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / VOICE_MODEL
    if not out.exists():
        print(f"Downloading {VOICE_URL.format(VOICE_MODEL)}...")
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(VOICE_URL.format(VOICE_MODEL), tmp)
        tmp.replace(out)
    return out


def voices_dir(cfg: dict) -> Path:
    d = Path(cfg.get("data_dir") or "./data")
    return (d if d.is_absolute() else ROOT / d) / "voices"


class VoiceID:
    """Speaker embeddings for stretches of 16 kHz mono PCM."""

    min_s = 1.5     # shorter lines are widened to this, around their middle

    def __init__(self, cfg: dict):
        import sherpa_onnx
        p = model_path(cfg)
        if not p.exists():
            raise RuntimeError(f"no voice model at {p} (run: python -m clipfarm setup-speech --clip)")
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(p), num_threads=int(cfg.get("threads", 2))))

    def embed(self, pcm: bytes, start: float, end: float,
              sample_rate: int = SAMPLE_RATE) -> list[float] | None:
        mid, half = (start + end) / 2, max((end - start) / 2, self.min_s / 2)
        a = max(0, int((mid - half) * sample_rate)) * 2
        b = int((mid + half) * sample_rate) * 2
        x = [v / 32768 for v in samples(pcm[a:b])]
        if len(x) < sample_rate // 2:
            return None
        s = self.ex.create_stream()
        s.accept_waveform(sample_rate, x)
        s.input_finished()
        return _unit(self.ex.compute(s))


def _unit(v) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VoicePrint:
    """A streamer's averaged embedding, saved as data/voices/<channel>.json."""

    def __init__(self, path: Path, embedding: list[float] | None = None,
                 seconds: float = 0.0, takes: list | None = None):
        self.path, self.embedding, self.seconds = path, embedding, seconds
        self.takes = takes or []

    @classmethod
    def load(cls, cfg: dict, channel: str) -> "VoicePrint":
        path = voices_dir(cfg) / f"{channel.lower()}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(path, d.get("embedding"), float(d.get("seconds", 0)), d.get("takes"))
        return cls(path)

    def add(self, emb: list[float], seconds: float, note: str = "") -> None:
        if self.embedding is None:
            self.embedding = emb
        else:                                   # average weighted by talking time
            w = self.seconds / (self.seconds + seconds)
            self.embedding = _unit([w * a + (1 - w) * b for a, b in zip(self.embedding, emb)])
        self.seconds += seconds
        self.takes.append({"note": note, "seconds": round(seconds, 1),
                           "at": time.strftime("%Y-%m-%d %H:%M")})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"embedding": self.embedding, "seconds": self.seconds,
                                         "takes": self.takes}), encoding="utf-8")


def sentences(words: list[Word], pause_s: float = 0.35) -> list[list[Word]]:
    """Words split where a sentence ends or the speaker pauses (voices take turns there)."""
    out, cur = [], []
    for w in words:
        if cur and (w.start - cur[-1].end > pause_s or cur[-1].text.endswith(SENTENCE_END)):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def tag_voices(words: list[Word], pcm: bytes, offset: float, vid: VoiceID,
               print_emb: list[float]) -> list[Word]:
    """Each word gets its sentence's similarity to the streamer (Word.voice). Word times
    are offset by `offset` into the PCM (the raw clip's lead)."""
    out = []
    for sent in sentences(words):
        emb = vid.embed(pcm, sent[0].start + offset, sent[-1].end + offset)
        sim = None if emb is None else round(similarity(emb, print_emb), 3)
        out += [replace(w, voice=sim) for w in sent]
    return out


def streamer_only(words: list[Word], min_sim: float) -> list[Word]:
    return [w for w in words if w.voice is None or w.voice >= min_sim]


def enrol(cfg: dict, channel: str, pcm: bytes, start: float, end: float,
          note: str = "") -> VoicePrint:
    """Add a stretch where only the streamer talks to their voiceprint."""
    if end - start < 3:
        raise ValueError("use at least 3 s of the streamer talking (10-20 s is better)")
    vp = VoicePrint.load(cfg, channel)
    emb = VoiceID(cfg.get("voice") or {}).embed(pcm, start, end)
    if emb is None:
        raise ValueError("that stretch has no audio")
    vp.add(emb, end - start, note)
    vp.save()
    return vp
