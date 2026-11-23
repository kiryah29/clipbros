"""ClipAgent: combination rules, clip types, confidence floor, candidate JSON, hooks."""
import json

from clipfarm.agent import below_floor, candidate_json, chat_evidence, evaluate, speech_evidence
from clipfarm.chat.simulated import SimulatedChat
from clipfarm.cli import ranked_candidates
from clipfarm.config import channel_settings, load_settings
from clipfarm.engine import make_detector, save_receipt, score_moment
from clipfarm.hooks import write_hook
from clipfarm.models import Clip

S = load_settings()
CFG = channel_settings(S, {"name": "jasontheween", "platform": "twitch", "streamer": "Jason"})
GTA = channel_settings(S, {"name": "jasontheween", "platform": "twitch", "game": "gta6",
                           "streamer": "Jason"})
AG = CFG["clip_agent"]


def words(text, start=0.0):
    return [[w, start + i * 0.3, start + i * 0.3 + 0.25] for i, w in enumerate(text.split())]


def receipt(msgs=(), chat_x=1.0, audio_x=1.0, said="", spike=True, moment=None, cfg=CFG):
    r = {"chat_ratio": chat_x, "audio_ratio": audio_x, "transcript": words(said),
         "chat_evidence": chat_evidence(msgs, cfg["clip_agent"]) if msgs else None,
         "moment": moment, "speech": {}}
    if spike:
        r.update(peak_rate=10.0, ratio=chat_x)
    return r


def test_one_pog_is_weak_fifty_is_a_bucket():
    ce = chat_evidence(["Pog"] + ["hi"] * 49, AG)
    assert ce["buckets"]["hype"]["share"] == 0.02
    c = evaluate(receipt(["Pog"] + ["hi"] * 49, chat_x=1.0, spike=False), CFG, 5, 0, 30)
    assert not c["evidence"]["keywords_detected"] and not c["recommended"]
    ce = chat_evidence(["POGGERS"] * 30 + ["LETSGOOOO !!!"] * 20, AG)
    assert ce["buckets"]["hype"]["share"] == 1.0
    assert ce["punct_share"] == 0.4 and ce["emote_wall"]["share"] == 0.6


def test_lone_keyword_never_makes_a_clip():
    # chat full of KEKW and the streamer saying "no way", but no chat or audio spike
    r = receipt(["KEKW KEKW"] * 40, chat_x=1.0, said="no way that happened", spike=False)
    c = evaluate(r, CFG, 30, 0, 30)
    assert "chat_keywords" in c["signals_present"] and "transcript_trigger" in c["signals_present"]
    assert c["confidence_score"] < 0.65 and not c["recommended"]
    assert "keywords alone" in c["reason"]
    assert below_floor({"clip_agent": c}, CFG)


def test_single_signal_spike_is_blocked():
    c = evaluate(receipt(chat_x=1.0, audio_x=8.0, spike=False), CFG, 35, 0, 30)
    assert c["signals_present"] == ["audio_spike"] and not c["recommended"]
    assert "only one signal" in c["reason"]


def test_combination_is_recommended_and_typed_funny():
    msgs = ["KEKW"] * 25 + ["OMEGALUL !!"] * 15 + ["lol"] * 10
    r = receipt(msgs, chat_x=6.0, audio_x=4.0, said="oh my god bro")
    c = evaluate(r, CFG, 70, 1000, 1030, origin_ts=0)
    assert c["clip_type"] == "funny" and c["recommended"] and c["confidence_score"] >= 0.8
    assert c["timestamp_start"] == "00:16:40" and c["timestamp_end"] == "00:17:10"
    assert set(c["evidence"]["keywords_detected"]) >= {"KEKW", "OMEGALUL"}
    assert c["evidence"]["chat_velocity_spike"] and c["evidence"]["audio_spike"]
    assert c["evidence"]["transcript_trigger"] == "oh my god"


def test_clip_types_from_buckets_phrases_and_context():
    clutch = evaluate(receipt(["CLUTCH"] * 20 + ["EZClap"] * 10, chat_x=5, said="let's go"),
                      CFG, 60, 0, 30)
    fail = evaluate(receipt(["RIP"] * 30 + ["F"] * 10, chat_x=5), CFG, 60, 0, 30)
    rage = evaluate(receipt(["?"] * 20, chat_x=4, audio_x=5,
                            said="bruh i'm done alt f4 what the hell"), CFG, 60, 0, 30)
    donation = evaluate(receipt(["LUL"] * 5 + ["hi"] * 20, chat_x=4,
                                said="thank you for the hundred gifted"), CFG, 60, 0, 30)
    talk = evaluate(receipt(["L"] * 20, chat_x=4, said="wait chat said what"), CFG, 60, 0, 30)
    assert [x["clip_type"] for x in (clutch, fail, rage, donation, talk)] == \
        ["clutch", "fail", "rage", "donation", "chat_interaction"]
    assert "donation detected" in donation["evidence"]["context_note"]
    assert "context_event" in rage["signals_present"]


def test_conflicting_signals_lower_confidence():
    msgs = ["KEKW"] * 20 + ["hi"] * 20
    calm = evaluate(receipt(msgs, chat_x=5, said="oh my god"), CFG, 55, 0, 30)
    mixed = evaluate(receipt(msgs, chat_x=5, said="oh my god alt f4 i quit"), CFG, 55, 0, 30)
    assert mixed["conflicts"] and mixed["confidence_score"] < calm["confidence_score"]
    loud_quiet = evaluate(receipt(chat_x=1.0, audio_x=8, said="no way", spike=False),
                          CFG, 45, 0, 30)
    assert any("chat stayed" in x for x in loud_quiet["conflicts"])


def test_keywords_adapt_per_game_and_channel():
    # GTA profile adds WASTED to outcome_loss on top of the default RIP
    assert "outcome_loss" not in chat_evidence(["WASTED"] * 10, AG)["buckets"]
    assert chat_evidence(["WASTED"] * 10, GTA["clip_agent"])["buckets"]["outcome_loss"]["msgs"] == 10
    ch = channel_settings(S, {"name": "x", "platform": "twitch",
                              "clip_agent": {"extra_chat_keywords": {"funny": ["jasonLUL"]}}})
    assert chat_evidence(["jasonLUL"] * 5, ch["clip_agent"])["buckets"]["funny"]["msgs"] == 5
    assert ch["clip_agent"]["min_confidence"] == 0.65            # rest of the defaults kept


def test_speech_buckets_collapse_stretched_words():
    assert speech_evidence(words("LETS GOOOOO we did it"), AG)["celebration"] == ["lets go"]
    assert speech_evidence(words("are you kidding me"), AG)["shock"] == ["are you kidding me"]


def test_detector_attaches_chat_evidence_and_moment_type():
    chat = SimulatedChat("t", minutes=8, bursts=[(300, 10, 8)], start_ts=0,
                         hype_lines=["COPS 💀", "5 STAR LMAO", "the police 😭"])
    det = make_detector("t", GTA)
    last = 0
    for m in chat.generate():
        det.feed(m)
        last = m.ts
    det.tick(last + 60)
    sp = det.spikes[0]
    assert sp.chat_evidence["messages"] > 20 and sp.chat_evidence["buckets"]
    win = dict(start=sp.peak_ts - 20, end=sp.end_ts + 5, triggers=["chat"], spike=sp, audio=None)
    score, r = score_moment(win, GTA, det, None, None, origin_ts=0)
    ca = r["clip_agent"]
    assert ca["evidence"]["chat_velocity_spike"] and ca["clip_type"] in ("funny", "hype")
    assert r["moment"]["clip_type"] == "hype"           # police_escalation in gta6.yaml


def test_candidate_json_saved_next_to_receipt(tmp_path):
    r = receipt(["KEKW"] * 30, chat_x=6, audio_x=4, said="no way")
    r["clip_agent"] = evaluate(r, CFG, 70, 0, 30)
    rpath = tmp_path / "clips" / "x" / "00001_receipt.json"
    save_receipt(rpath, r)
    cand = json.loads((rpath.parent / "00001_candidate.json").read_text(encoding="utf-8"))
    assert list(cand) == ["timestamp_start", "timestamp_end", "clip_type", "confidence_score",
                          "evidence", "reason"]
    assert set(cand["evidence"]) == {"keywords_detected", "chat_velocity_spike", "audio_spike",
                                     "transcript_trigger", "context_note"}
    assert 0 <= cand["confidence_score"] <= 1
    assert json.loads(rpath.read_text())["clip_agent"]["recommended"]


def test_ranked_candidates_by_confidence_and_floor():
    def clip(i, conf, score):
        ca = {"timestamp_start": "00:00:00", "timestamp_end": "00:00:30", "clip_type": "hype",
              "confidence_score": conf, "evidence": {}, "reason": "", "recommended": conf >= .65}
        return Clip(i, "x", "twitch", 0, 30, score, "recorded", None, None, {"clip_agent": ca}, 0)
    clips = [clip(1, 0.7, 90), clip(2, 0.9, 40), clip(3, 0.4, 99)]
    assert [c["clip_id"] for c in ranked_candidates(clips)] == [2, 1]
    assert [c["clip_id"] for c in ranked_candidates(clips, True)] == [2, 1, 3]


def test_clip_type_steers_the_hook():
    base = {"triggers": ["chat"], "chat_ratio": 4.0, "ratio": 4.0, "audio_ratio": 1.0,
            "top_tokens": [("KEKW", 40)], "speech": {}}
    plain = write_hook(base, CFG, seed=3)
    donation = write_hook({**base, "clip_agent": {"clip_type": "donation"}}, CFG, seed=3)
    clutch = write_hook({**base, "clip_agent": {"clip_type": "clutch"}}, CFG, seed=3)
    assert plain[1] == "laugh"
    assert donation[1] == "type_donation" and "donation" in donation[0].lower()
    assert clutch[1] == "type_clutch"
    # hype/funny come after chat's own reaction, so they don't override it
    assert write_hook({**base, "clip_agent": {"clip_type": "funny"}}, CFG, seed=3)[1] == "laugh"


def test_gate_can_be_turned_off():
    c = {"recommended": False, "confidence_score": 0.3, "reason": "x"}
    off = {**CFG, "clip_agent": {**AG, "gate": False}}
    assert below_floor({"clip_agent": c}, CFG) and below_floor({"clip_agent": c}, off) is None
    assert candidate_json({**c, "timestamp_start": 1, "timestamp_end": 2, "clip_type": "x",
                           "evidence": {}})["confidence_score"] == 0.3
