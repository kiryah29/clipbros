"""Captions keep the streamer's voice and drop game characters; hooks only quote the streamer."""
from pathlib import Path

import pytest

from clipfarm import engine
from clipfarm.config import ROOT, channel_settings, load_settings
from clipfarm.craft.captions import Word
from clipfarm.voice import VOICE_MODEL, VoicePrint, sentences, similarity, streamer_only

S = load_settings()


def cfg_in(tmp_path):
    return {**channel_settings(S, {"name": "nick", "platform": "twitch", "streamer": "Nick"}),
            "data_dir": str(tmp_path)}


def test_sentences_split_on_punctuation_and_pauses():
    ws = [Word("Where's", 0, 0.3), Word("the", 0.3, 0.4), Word("safe?", 0.4, 0.8),
          Word("Receiving", 0.9, 1.3), Word("reports", 1.3, 1.8), Word("no", 3.0, 3.2)]
    assert [[w.text for w in s] for s in sentences(ws)] == \
        [["Where's", "the", "safe?"], ["Receiving", "reports"], ["no"]]


def test_voiceprint_averages_takes_and_survives_a_reload(tmp_path):
    cfg = cfg_in(tmp_path)
    vp = VoicePrint.load(cfg, "Nick")
    assert vp.embedding is None
    vp.add([1.0, 0.0], 10, "a")
    vp.add([0.0, 1.0], 10, "b")
    vp.save()
    again = VoicePrint.load(cfg, "nick")                   # channel names are case-insensitive
    assert again.seconds == 20 and len(again.takes) == 2
    assert similarity(again.embedding, [1.0, 0.0]) == pytest.approx(similarity(
        again.embedding, [0.0, 1.0]))


class FakeVoices:
    """Embeds by time: the streamer talks before 10 s, the police radio after."""

    def embed(self, pcm, start, end, sample_rate=16000):
        return [1.0, 0.0] if start < 10 else [0.0, 1.0]


def test_lines_in_another_voice_are_left_out(tmp_path, monkeypatch):
    cfg = cfg_in(tmp_path)
    vp = VoicePrint.load(cfg, "nick")
    vp.add([1.0, 0.0], 12)
    vp.save()
    monkeypatch.setattr(engine, "voice_id", lambda vcfg: FakeVoices())
    words = [Word("Yeah,", 2, 2.3), Word("where's", 2.3, 2.6), Word("the", 2.6, 2.7),
             Word("safe?", 2.7, 3.0), Word("Receiving", 12, 12.4), Word("reports.", 12.4, 13)]
    kept, report, _ = engine.keep_streamer(words, "nick", cfg, Path("x.mp4"), 0.0, pcm=b"")
    assert [w.text for w in kept] == ["Yeah,", "where's", "the", "safe?"]
    assert report["dropped"] == [["Receiving reports.", 0.0]]
    assert all(w.voice == 1.0 for w in kept)


def test_no_voiceprint_changes_nothing(tmp_path):
    words = [Word("hi", 0, 1)]
    kept, report, _ = engine.keep_streamer(words, "nobody", cfg_in(tmp_path), Path("x.mp4"), 0)
    assert kept == words and report == {"enrolled": False}


def test_hooks_only_quote_the_streamer(tmp_path):
    cfg = cfg_in(tmp_path)
    receipt = {"triggers": ["chat"], "chat_ratio": 4.0, "ratio": 4.0, "audio_ratio": 2.0,
               "top_tokens": [], "speech": {"phrases": []}, "hook": '"x"', "hook_kind": "quote",
               "peak_ts": 5}
    npc = [Word("I'm", 1.0, 1.2, 0.05), Word("calling", 1.2, 1.5, 0.05), Word("about", 1.5, 1.8, 0.05),
           Word("Lucia", 1.8, 2.1, 0.05), Word("Caminos.", 2.1, 2.6, 0.05)]
    hook, kind = engine.refresh_hook(receipt, cfg, npc)
    assert kind != "quote" and "Caminos" not in hook
    nick = [replace_voice(w, 0.6) for w in npc]
    assert engine.refresh_hook(receipt, cfg, nick)[1] == "quote"


def replace_voice(w, v):
    return Word(w.text, w.start, w.end, v)


@pytest.mark.skipif(not (ROOT / "models" / VOICE_MODEL).exists(),
                    reason="voice model not downloaded (setup-speech --clip)")
def test_real_model_embeds_audio():
    from clipfarm.voice import VoiceID
    import math
    pcm = b"".join(int(8000 * math.sin(i / 8)).to_bytes(2, "little", signed=True)
                   for i in range(16000 * 2))
    e = VoiceID({}).embed(pcm, 0.2, 1.8)
    assert e and abs(similarity(e, e) - 1) < 1e-3
