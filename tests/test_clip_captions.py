"""Captions and hook quotes come from a second transcription of the cut clip, quotes are
whole sentences said early in the reel, and sound tags never reach the screen."""
from pathlib import Path

import pytest

from clipfarm import engine
from clipfarm.config import ROOT, channel_settings, load_settings
from clipfarm.craft.captions import Word, drop_sound_tags
from clipfarm.hooks import quote_near, sentences
from clipfarm.queue import ClipQueue
from clipfarm.speech import CLIP_MODEL, make_clip_transcriber, token_words

S = load_settings()
CFG = channel_settings(S, {"name": "nickmercs", "platform": "twitch", "streamer": "Nick"})


def w(text, start, end):
    return [text, start, end]


def test_tokens_become_words_with_their_own_times():
    words = token_words([" Oh", " my", " go", "sh", "."], [1.0, 1.3, 1.6, 1.7, 1.9], offset=10)
    assert [x.text for x in words] == ["Oh", "my", "gosh."]
    assert [round(x.start, 2) for x in words] == [11.0, 11.3, 11.6]
    assert words[0].end <= words[1].start and words[-1].end > words[-1].start


def test_quotes_are_whole_sentences_never_fragments():
    # the window cuts into a sentence: "...if you can come back have" must not be quoted
    cut = [w("if", 0.0, 0.2), w("you", 0.2, 0.3), w("can", 0.3, 0.5), w("come", 0.5, 0.7),
           w("back", 0.7, 0.9), w("have", 0.9, 1.1)]
    assert quote_near(cut, 1.5) == ""
    talk = [w("so", 0.0, 0.2), w("yeah.", 0.2, 0.4), w("Who's", 1.0, 1.3), w("the", 1.3, 1.4),
            w("dreadlock", 1.4, 1.9), w("homie?", 1.9, 2.3), w("Don't", 2.5, 2.7),
            w("worry,", 2.7, 3.0)]
    assert quote_near(talk, 3.0) == "Who's the dreadlock homie?"
    # no punctuation at all: a pause still ends the sentence
    paused = [w("watch", 2.0, 2.3), w("me", 2.3, 2.5), w("steal", 2.5, 2.9), w("this", 2.9, 3.1),
              w("car", 3.1, 3.5), w("later", 6.0, 6.4)]
    assert quote_near(paused, 4.0) == "watch me steal this car"
    assert [len(s) for s in sentences(talk)] == [4]          # "so yeah." starts mid-sentence


def test_sound_tags_and_lyrics_are_not_captioned():
    words = [Word("*SCREAMS*", 0, 1), Word("oh", 1, 1.2), Word("[MUSIC", 1.2, 1.4),
             Word("PLAYING]", 1.4, 1.6), Word("♪", 1.6, 1.7), Word("no", 1.7, 2),
             Word("(laughs)", 2, 2.2), Word("way!", 2.2, 2.5)]
    assert [x.text for x in drop_sound_tags(words)] == ["oh", "no", "way!"]


def phrase_receipt(hook='"LET\'S GO" - Nick'):
    return {"triggers": ["chat"], "chat_ratio": 6.0, "ratio": 6.0, "audio_ratio": 2.5,
            "top_tokens": [("LUL", 7), ("lol", 2)], "speech": {"phrases": ["let's go"]},
            "hook": hook, "hook_kind": "phrase", "peak_ts": 100}


def test_phrase_hook_is_dropped_when_the_clip_never_says_it():
    words = [Word("Hello,", 0.5, 0.8), Word("bro.", 0.8, 1.0), Word("Now", 3.0, 3.2),
             Word("you", 3.2, 3.3), Word("listen.", 3.3, 3.7)]
    hook, kind = engine.refresh_hook(phrase_receipt(), CFG, words)
    assert kind != "phrase" and "LET'S GO" not in hook.upper()


def test_phrase_hook_said_too_late_is_dropped_but_early_one_kept():
    late = [Word("let's", 25.0, 25.3), Word("go!", 25.3, 25.6)]
    assert engine.refresh_hook(phrase_receipt(), CFG, late)[1] != "phrase"
    early = [Word("let's", 2.0, 2.3), Word("go!", 2.3, 2.6)]
    hook, kind = engine.refresh_hook(phrase_receipt(), CFG, early)
    assert kind == "phrase" and "LET'S GO" in hook.upper()


def test_other_hooks_are_left_alone():
    r = {**phrase_receipt(), "hook": "Chat lost it at this", "hook_kind": "laugh"}
    assert engine.refresh_hook(r, CFG, [Word("x", 0, 1)]) == ("Chat lost it at this", "laugh")


def test_craft_uses_the_clip_transcript_and_rechecks_the_hook(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(engine, "render_reel",
                        lambda *a, **k: seen.update(k, words=a[4]))
    monkeypatch.setattr(engine, "pick_facecam", lambda *a, **k: (None, "none"))
    monkeypatch.setattr(engine, "decode_pcm", lambda p: b"")
    monkeypatch.setattr(engine, "clip_words", lambda raw, cfg, lead: [
        Word("*MUSIC*", 0.1, 0.3), Word("Dude,", 0.5, 0.8), Word("dude,", 0.8, 1.0),
        Word("dude.", 1.0, 1.3), Word("No", 1.5, 1.7), Word("way.", 1.7, 2.0)])
    q = ClipQueue(tmp_path / "q.db")
    live = [["LET'S", 0.4, 0.6], ["GO", 0.6, 0.9]]
    cid = q.add("nickmercs", "twitch", 0, 30, 50,
                {**phrase_receipt(), "transcript": live}, raw_path=str(tmp_path / "r.mp4"))
    clean = lambda p, c, **k: {"decision": "clean", "ranges": [], "why": "t"}   # noqa: E731
    engine.craft_clip(q, cid, CFG, None, nudity_check=clean)
    assert [x.text for x in seen["words"]][:2] == ["Dude,", "dude,"]      # tag dropped
    r = q.get(cid).receipt
    assert r["caption_source"] == "clip" and r["hook_live"] == '"LET\'S GO" - Nick'
    assert "LET'S GO" not in r["hook"].upper() and seen["hook_text"] == r["hook"]


def test_craft_falls_back_to_the_live_transcript_without_the_model(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(engine, "render_reel", lambda *a, **k: seen.update(words=a[4]))
    monkeypatch.setattr(engine, "pick_facecam", lambda *a, **k: (None, "none"))
    monkeypatch.setattr(engine, "decode_pcm", lambda p: b"")
    monkeypatch.setattr(engine, "clip_words", lambda *a: None)
    q = ClipQueue(tmp_path / "q.db")
    cid = q.add("x", "twitch", 0, 30, 50, {"transcript": [["hi", 0.5, 0.8]]},
                raw_path=str(tmp_path / "r.mp4"))
    clean = lambda p, c, **k: {"decision": "clean", "ranges": [], "why": "t"}   # noqa: E731
    engine.craft_clip(q, cid, CFG, None, nudity_check=clean)
    assert [x.text for x in seen["words"]] == ["hi"]
    assert q.get(cid).receipt["caption_source"] == "live"


@pytest.mark.skipif(not (ROOT / "models" / CLIP_MODEL).exists(),
                    reason="caption model not downloaded (setup-speech --clip)")
def test_parakeet_transcribes_its_sample():
    asr = make_clip_transcriber(S["speech"])
    wav = next((ROOT / "models" / CLIP_MODEL).glob("test_wavs/*.wav"), None)
    if wav is None:
        pytest.skip("no sample wav shipped with the model")
    words = asr.transcribe_file(Path(wav))
    assert len(words) > 3 and all(a.start <= b.start for a, b in zip(words, words[1:]))
