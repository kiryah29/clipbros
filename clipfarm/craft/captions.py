"""Word-level captions and the CTA rendered as one ASS subtitle file.

The hook card is drawn separately as a brand overlay (see overlay.py)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe(media: Path, engine: str) -> list[Word]:
    """Word timestamps from the clip audio. 'none' returns no captions."""
    if engine == "faster_whisper":
        from faster_whisper import WhisperModel  # optional dependency
        model = WhisperModel("small", compute_type="int8")
        segments, _ = model.transcribe(str(media), word_timestamps=True, vad_filter=True)
        return [Word(w.word.strip(), w.start, w.end) for s in segments for w in s.words]
    return []


def fake_words(text: str, start: float, word_s: float = 0.32) -> list[Word]:
    """Evenly timed words; used by demo mode in place of a real transcript."""
    out, t = [], start
    for w in text.split():
        out.append(Word(w, t, t + word_s))
        t += word_s
    return out


STYLES = {
    # name: (primary, highlight, outline, back, border_style, scale_active)
    "pop":   ("&H00FFFFFF", "&H0000E5FF", "&H00000000", "&H00000000", 1, 115),
    "boxed": ("&H00FFFFFF", "&H00FFFFFF", "&H00000000", "&H000000FF", 3, 100),
    "clean": ("&H00FFFFFF", "&H00FFFFFF", "&H64000000", "&H00000000", 1, 100),
    # ClipBros mockup: bold white caps, the spoken word in lime (#C8F032), dark outline
    "clipbros": ("&H00FFFFFF", "&H0032F0C8", "&H00101010", "&H80000000", 1, 104),
}


def _t(s: float) -> str:
    s = max(0.0, s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(words: list[Word], duration: float, cfg: dict, caption_y: int, out: Path,
              chunk: int = 3, caption_x: int | None = None, max_w: int | None = None) -> Path:
    W, H = cfg["width"], cfg["height"]
    cx = caption_x if caption_x is not None else W // 2
    max_w = max_w or W - 120
    prim, hi, outl, back, border, scale = STYLES.get(cfg["caption_style"], STYLES["pop"])
    font = cfg.get("caption_font", "Inter ExtraBold")
    max_fs = int(cfg.get("caption_size", 84))
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
        "WrapStyle: 2", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,{font},{max_fs},{prim},{hi},{outl},{back},1,0,0,0,100,100,0,0,{border},6,3,5,60,60,0,1",
        f"Style: CTA,{font},64,&H00111111,&H00111111,&H00EAF1F3,&H00EAF1F3,1,0,0,0,100,100,0,0,3,20,0,2,80,80,420,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # captions: show `chunk` words at a time, active word highlighted
    for i in range(0, len(words), chunk):
        group = words[i:i + chunk]
        for j, w in enumerate(group):
            end = group[j + 1].start if j + 1 < len(group) else w.end
            parts = []
            for k, g in enumerate(group):
                txt = _esc(g.text.upper())
                if k == j:
                    parts.append(f"{{\\c{hi}\\fscx{scale}\\fscy{scale}}}{txt}{{\\r}}")
                else:
                    parts.append(txt)
            n_chars = len(" ".join(g.text for g in group))
            fs = min(max_fs, int(max_w / (0.70 * max(n_chars, 1))))  # shrink long chunks to fit
            lines.append(f"Dialogue: 1,{_t(w.start)},{_t(end)},Caption,,0,0,0,,"
                         f"{{\\pos({cx},{caption_y})\\fs{fs}}}" + " ".join(parts))
    if cfg.get("cta_text"):
        lines.append(f"Dialogue: 2,{_t(duration - cfg['cta_seconds'])},{_t(duration)},CTA,,0,0,0,,"
                     f"{{\\fad(200,0)}}{_esc(cfg['cta_text'])}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
