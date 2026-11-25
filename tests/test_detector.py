from clipfarm.chat.simulated import SimulatedChat
from clipfarm.config import load_settings
from clipfarm.detector import SpikeDetector, score_spike
from clipfarm.engine import clip_window

S = load_settings()


def run(chat):
    det = SpikeDetector(chat.channel, S["detector"], S["ranking"]["hype_tokens"])
    last = 0
    for m in chat.generate():
        det.feed(m)
        last = m.ts
    det.tick(last + 60)
    return det


def test_detects_each_injected_burst():
    bursts = [(200, 10, 8), (420, 8, 6), (640, 12, 10)]
    chat = SimulatedChat("t", minutes=12, bursts=bursts, start_ts=0)
    det = run(chat)
    assert len(det.spikes) == 3
    for sp, (b, dur, _) in zip(det.spikes, bursts):
        assert b - 2 <= sp.peak_ts <= b + dur + 4


def test_no_spikes_on_flat_chat():
    chat = SimulatedChat("t", minutes=15, bursts=[], start_ts=0)
    assert run(chat).spikes == []


def test_no_spike_during_warmup():
    chat = SimulatedChat("t", minutes=5, bursts=[(30, 10, 10)], start_ts=0)
    assert run(chat).spikes == []


def test_baseline_adapts_to_busier_channel():
    quiet = run(SimulatedChat("q", base_rate=1, minutes=8, bursts=[], start_ts=0))
    busy = run(SimulatedChat("b", base_rate=10, minutes=8, bursts=[], start_ts=0))
    assert busy.mean > quiet.mean * 5
    assert busy.spikes == [] and quiet.spikes == []


def test_bigger_burst_scores_higher_and_receipt_filled():
    chat = SimulatedChat("t", minutes=10, bursts=[(200, 10, 4), (450, 10, 12)], start_ts=0)
    small, big = run(chat).spikes
    assert score_spike(big, S["signals"]["weights"]) > score_spike(small, S["signals"]["weights"])
    assert big.hype_share > 0.3 and big.top_tokens and big.rate_curve


def test_clip_window_looks_back_and_caps_length():
    chat = SimulatedChat("t", minutes=6, bursts=[(200, 10, 8)], start_ts=0)
    sp = run(chat).spikes[0]
    start, end = clip_window(sp, S["clip"])
    assert start < sp.peak_ts < end
    assert end - start <= S["clip"]["raw_max_s"]   # the reel itself is anchored inside
