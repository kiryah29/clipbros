"""Word-level captions and the CTA rendered as one ASS subtitle file.

The hook card is drawn separately as a brand overlay (see overlay.py)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .. import brand


@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe(media: Path, engine: str) -> list[Word]:
    """Word timestamps from the clip audio. 'none' returns no captions; 'auto' uses
    faster-whisper when it is installed (live runs pass the stream transcript instead);
    'clip' uses Parakeet (speech.Parakeet), see engine.clip_words."""
    if engine == "auto":
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return []
        engine = "faster_whisper"
    if engine == "faster_whisper":
        from faster_whisper import WhisperModel  # optional dependency
        model = WhisperModel("small", compute_type="int8")
        segments, _ = model.transcribe(str(media), word_timestamps=True, vad_filter=True)
        return [Word(w.word.strip(), w.start, w.end) for s in segments for w in s.words]
    return []


def drop_sound_tags(words: list[Word]) -> list[Word]:
    """No "*SCREAMS*", "[MUSIC]", "(laughs)" or "♪" lyric marks on screen: speech models
    write these for sounds, and the reviewed reels showed them as captions."""
    out, inside = [], None
    for w in words:
        t = w.text.strip()
        if inside:                                  # a tag spread over several words
            if t.rstrip(".,!?").endswith(inside):
                inside = None
            continue
        if "♪" in t:
            continue
        opener = next((c for o, c in (("*", "*"), ("[", "]"), ("(", ")")) if t.startswith(o)), None)
        if opener:
            if not (len(t) > 1 and t.rstrip(".,!?").endswith(opener)):
                inside = opener
            continue
        out.append(w)
    return out


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
    # ClipBros brand kit: Inter 800 caps in Paper, Ink outline, the spoken word in Clip Lime
    "clipbros": (brand.ass_colour(brand.PAPER), brand.ass_colour(brand.LIME),
                 brand.ass_colour(brand.INK), brand.ass_colour(brand.INK, 0x80), 1, 100),
}


def _t(s: float) -> str:
    s = max(0.0, s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}"


# word roots masked anywhere in a word ("fucking" -> "f******"), and whole words only
CENSOR_ROOTS = ["fuck", "shit", "bitch", "cunt", "nigg", "pussy", "whore", "slut", "fag",
                "dick", "cock", "asshole", "bastard", "retard"]
CENSOR_WORDS = ["ass", "hoe", "hoes", "damn"]


def censor(word: str, roots=CENSOR_ROOTS, whole=CENSOR_WORDS) -> str:
    """Keep the first letter and punctuation, star the rest: "FUCKING," -> "F******,"."""
    letters = "".join(c for c in word.lower() if c.isalpha())
    if not letters or not (letters in whole or any(r in letters for r in roots)):
        return word
    seen, out = False, []
    for c in word:
        out.append(c if not c.isalpha() or not seen else "*")
        seen = seen or c.isalpha()
    return "".join(out)


def group_words(words: list[Word], size: int, gap_s: float) -> list[list[Word]]:
    """Caption lines of up to `size` words; a pause longer than `gap_s` ends a line."""
    groups: list[list[Word]] = []
    for w in words:
        if groups and len(groups[-1]) < size and w.start - groups[-1][-1].end <= gap_s:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def subtitle_times(media: Path, box: dict, fps: float = 2, width: int = 480,
                   height: int = 80) -> list[float]:
    """Seconds (in the clip) where the game shows its own subtitles inside `box`
    (fractions of the frame): bright text strokes on a dark background. GTA cutscene and
    NPC lines come with subtitles, so the speech model's transcript of them is game
    dialogue, not the streamer."""
    vf = (f"fps={fps},crop=iw*{box['w']}:ih*{box['h']}:iw*{box['x']}:ih*{box['y']},"
          f"scale={width}:{height},format=gray")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(media), "-vf", vf,
                        "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if not r.stdout:
        return []
    n = width * height
    times = []
    for k in range(len(r.stdout) // n):
        if looks_like_subtitle(r.stdout[k * n:(k + 1) * n], width):
            times.append(k / fps)
    return times


def looks_like_subtitle(frame: bytes, width: int, bright: int = 170, dark: int = 80,
                        reach: int = 3) -> bool:
    """Most bright pixels have dark pixels just left or right of them (thin text strokes).
    Measured on GTA 6 clips: subtitle frames 450-1100 such pixels at >85 % of all bright
    ones; gameplay (headlights, signs, sky) well under 40 %."""
    lit = edge = 0
    for r in range(len(frame) // width):
        row = frame[r * width:(r + 1) * width]
        for c in range(reach, width - reach):
            if row[c] > bright:
                lit += 1
                if min(row[c - reach], row[c + reach]) < dark:
                    edge += 1
    return edge >= 150 and edge >= 0.6 * lit


def drop_dialogue(words: list[Word], times: list[float], pad: float = 0.5) -> list[Word]:
    """Words said while game subtitles were on screen are dropped."""
    return [w for w in words if not any(w.start - pad <= t <= w.end + pad for t in times)]


def _esc(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(words: list[Word], duration: float, cfg: dict, caption_y: int, out: Path,
              chunk: int = 3, caption_x: int | None = None, max_w: int | None = None,
              extra_styles: list[str] = (), extra_events: list[str] = ()) -> Path:
    W, H = cfg["width"], cfg["height"]
    cx = caption_x if caption_x is not None else W // 2
    max_w = max_w or W - 120
    prim, hi, outl, back, border, scale = STYLES.get(cfg["caption_style"], STYLES["pop"])
    font = cfg.get("caption_font", brand.ONCLIP_FONT_NAME)
    max_fs = int(cfg.get("caption_size", 84))
    outline = int(cfg.get("caption_outline", 5))
    chunk = int(cfg.get("caption_words", chunk))
    if cfg.get("caption_censor", True):   # TikTok limits reach for on-screen swearing
        roots = cfg.get("censor_roots", CENSOR_ROOTS)
        whole = cfg.get("censor_words", CENSOR_WORDS)
        words = [Word(censor(w.text, roots, whole), w.start, w.end) for w in words]
    ink, paper = brand.ass_colour(brand.INK), brand.ass_colour(brand.PAPER)   # CTA: Ink on Paper
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
        "WrapStyle: 2", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,{font},{max_fs},{prim},{hi},{outl},{back},1,0,0,0,100,100,0,0,{border},{outline},2,5,60,60,0,1",
        f"Style: CTA,{font},64,{ink},{ink},{paper},{paper},1,0,0,0,100,100,0,0,3,20,0,2,80,80,420,1",
        *extra_styles,
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # captions: show `chunk` words at a time, active word highlighted. A pause starts a
    # new line and a word stays up at most `caption_max_word_s`, so a line never sits on
    # screen through a silence (one stayed up 12 s).
    gap = float(cfg.get("caption_gap_s", 1.0))
    max_word = float(cfg.get("caption_max_word_s", 1.5))
    for group in group_words(words, chunk, gap):
        for j, w in enumerate(group):
            end = group[j + 1].start if j + 1 < len(group) else w.end
            end = min(end, w.start + max_word)
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
    lines.extend(extra_events)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
