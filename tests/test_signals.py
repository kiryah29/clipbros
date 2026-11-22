import array
import math
import subprocess
import sys

import pytest

from clipfarm.audio import AudioTracker, decode_pcm, loudness_per_second
from clipfarm.chat.simulated import SimulatedChat
from clipfarm.config import ROOT, load_settings
from clipfarm.craft.captions import Word
from clipfarm.engine import MomentBook, audio_window, make_detector, score_moment
from clipfarm.recorder import _kill_tree
from clipfarm.signals import BaselineTrigger, check_weights, score_signals, strength
from clipfarm.speech import SpeechTracker, make_transcriber, spread_words

S = load_settings()
CFG = {**S, "moments": [], "streamer": "t", "game_name": ""}


def test_weights_from_settings():
    w = S["signals"]["weights"]
    assert w["audio"] == 0.35 and w["speech"] == 0.20
    assert math.isclose(sum(w.values()), 1.0)
    with pytest.raises(ValueError):
        check_weights({**w, "audio": 0.5})
    with pytest.raises(ValueError):
        check_weights({"volume": 1.0})


def test_strength_scale():
    assert strength(1.0) == 0 and strength(0.5) == 0
    assert strength(3.0) == 0.5 and strength(5.0) == 1.0 and strength(50) == 1.0


def test_score_is_weighted_sum():
    score, pts = score_signals({"audio": 1.0, "speech": 0.5}, S["signals"]["weights"])
    assert score == 45.0 and pts["audio"] == 35.0 and pts["speech"] == 10.0


def test_baseline_trigger_fires_on_3x_only_after_warmup():
    t = BaselineTrigger(warmup_s=30, min_ratio=3, cooldown_s=0)
    levels = [0.05] * 20 + [0.5] * 3 + [0.05] * 40 + [0.3] * 3 + [0.05] * 10 + [0.15] * 3
    for i, v in enumerate(levels + [0.05] * 5):
        t.feed(i, v)
    # the warm-up blast never fires, 4x fires, ~2x does not
    assert len(t.events) == 1
    ev = t.events[0]
    assert ev["start_ts"] == 63 and ev["ratio"] >= 3.9
    assert t.peak_ratio(60, 70) >= 3.9 and t.peak_ratio(0, 10) == 1.0


def _tone(seconds: float, amp: float, sr: int = 16000) -> bytes:
    return array.array("h", (int(amp * 32767 * math.sin(2 * math.pi * 220 * i / sr))
                             for i in range(int(seconds * sr)))).tobytes()


def test_loudness_per_second():
    lv = loudness_per_second(_tone(1, 0.1) + _tone(1, 0.8))
    assert len(lv) == 2
    assert lv[0] == pytest.approx(0.1 / math.sqrt(2), rel=0.05)
    assert lv[1] == pytest.approx(0.8 / math.sqrt(2), rel=0.05)


def test_audio_tracker_on_real_decoded_audio(tmp_path):
    # quiet talk-level tone, then a 3 s "scream" at 8x, through ffmpeg like live segments
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "sine=f=300:d=40,volume='if(between(t,32,35),0.8,0.1)':eval=frame",
                    str(wav)], check=True)
    events = []
    tr = AudioTracker({**S["audio"], "warmup_s": 20}, on_event=events.append)
    pcm = decode_pcm(wav)
    tr.feed_pcm(1000.0, pcm)
    assert len(events) == 1 and 1031 <= events[0]["peak_ts"] <= 1035
    assert tr.ratio_between(1025, 1040) > 6
    start, end = audio_window(events[0], S["audio"], S["clip"])
    assert start < events[0]["peak_ts"] < end


def test_speech_phrases_pace_and_captions():
    sp = SpeechTracker(S["speech"])
    for i in range(30):                                    # normal pace: 1.5 words/s
        sp.add_chunk(i * 4.0, 4.0, spread_words("so we go here and then", 0, 4))
    sp.add_chunk(120.0, 4.0, spread_words(
        "no way no way oh my god let's go go go chat did you see that what the hell", 0, 4))
    hits, found = sp.phrase_hits(120, 124)
    assert hits >= 5 and "oh my god" in found and "go x3" in found
    val, info = sp.score(118, 126)
    assert info["pace_ratio"] > 2.5 and val > 0.7
    calm, _ = sp.score(40, 50)
    assert calm < 0.1
    words = sp.words_between(120, 124)
    assert words[0].text == "no" and words[0].start == pytest.approx(0)


def test_transcriber_off_or_missing_is_none():
    assert make_transcriber({"engine": "none"}) is None
    assert make_transcriber({"engine": "sherpa_onnx", "sherpa_model_dir": "/nope"},
                            quiet=True) is None


def test_moment_book_merges_scream_and_chat_reaction():
    book = MomentBook({**S["clip"], "max_length_s": 60})
    a, new = book.add(100, 120, "audio")
    b, new2 = book.add(95, 130, "chat")
    assert new and not new2 and a is b
    assert (a["start"], a["end"], a["triggers"]) == (95, 130, ["audio", "chat"])
    a["done"] = True
    c, _ = book.add(110, 135, "chat")       # overlaps a clip already cut: too little left
    assert c is None
    d, new3 = book.add(200, 230, "audio")
    assert new3 and len(book.items) == 2


def test_scream_plus_chat_outscores_chat_alone_and_audio_alone_counts():
    chat = SimulatedChat("t", minutes=8, bursts=[(300, 10, 8)], start_ts=0)
    det = make_detector("t", CFG)
    last = 0
    for m in chat.generate():
        det.feed(m)
        last = m.ts
    det.tick(last + 60)
    sp = det.spikes[0]
    audio = AudioTracker({**S["audio"], "warmup_s": 30})
    lv = [0.05] * 480
    for i in range(294, 298):
        lv[i] = 0.4                               # yell right before chat explodes
    for i in range(400, 403):
        lv[i] = 0.35                              # yell chat doesn't react to
    audio.feed_levels(0, lv)
    assert len(audio.trigger.events) == 2
    win = dict(start=sp.peak_ts - 20, end=sp.end_ts + 5, triggers=["chat"], spike=sp, audio=None)
    chat_only, _ = score_moment(win, CFG, det, None, None)
    both, r = score_moment(win, CFG, det, audio, None)
    assert both > chat_only + 25 and r["score_points"]["audio"] > 25
    assert "loudness" in r["why"]
    solo = dict(start=390, end=410, triggers=["audio"], spike=None, audio=audio.trigger.events[1])
    s_solo, r2 = score_moment(solo, CFG, det, audio, None)
    assert s_solo >= 30 and r2["why"].startswith("audio")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process check")
def test_kill_tree_stops_process():
    p = subprocess.Popen(["sleep", "60"])
    _kill_tree(p)
    assert p.poll() is not None
    _kill_tree(None)   # no-op


SHERPA = ROOT / "models" / "sherpa-onnx-whisper-tiny.en"


@pytest.mark.skipif(not SHERPA.exists(), reason="run: python -m clipfarm setup-speech")
def test_sherpa_transcribes_speech(tmp_path):
    pytest.importorskip("sherpa_onnx")
    wav = tmp_path / "say.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "sine=f=200:d=2", str(wav)], check=True)
    t = make_transcriber({**S["speech"], "engine": "sherpa_onnx"})
    assert t is not None
    words = t.transcribe_pcm(decode_pcm(wav))
    assert isinstance(words, list) and all(isinstance(w, Word) for w in words)
