import asyncio

from clipfarm import control
from clipfarm.control import RunManager
from clipfarm.streamers import StreamerRegistry


def test_record_is_idempotent_and_keeps_first_seen(tmp_path):
    reg = StreamerRegistry(tmp_path / "streamers.json")
    reg.record("JasonTheWeen", "twitch", "gta6", now=100)
    s = reg.record("jasontheween", "Twitch", now=200)
    assert len(reg.all()) == 1
    assert s == {"name": "jasontheween", "platform": "twitch", "first_seen": 100, "runs": 2,
                 "last_seen": 200, "game": "gta6"}
    reg.record("@SomeHandle", "youtube", now=300)
    assert [x["name"] for x in reg.all()] == ["somehandle", "jasontheween"]
    assert reg.names("twitch") == ["jasontheween"]
    # survives a restart
    assert StreamerRegistry(tmp_path / "streamers.json").names("youtube") == ["somehandle"]


def test_broken_file_starts_fresh(tmp_path):
    p = tmp_path / "streamers.json"
    p.write_text("{not json")
    reg = StreamerRegistry(p)
    assert reg.all() == []
    reg.record("caseoh_", "twitch")
    assert reg.names("twitch") == ["caseoh_"]


def test_control_run_saves_streamer_and_suggests_it(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load_channels", lambda: [{"name": "fromyaml", "platform": "twitch"}])
    reg = StreamerRegistry(tmp_path / "streamers.json")

    async def run():
        m = RunManager(tmp_path / "control", streamers=reg,
                       cmd=lambda d, ch, g, mins: ["python", "-c", "pass"])
        m.start("NewStreamer", "", 5)
        await asyncio.gather(*m.tasks)
        return m

    m = asyncio.run(run())
    assert reg.names("twitch") == ["newstreamer"]
    assert reg.all()[0]["source"] == "control"
    assert control.known_channels(reg) == ["fromyaml", "newstreamer"]
    # still saved after the run folder is cleaned up
    m.cleanup(now=10**12)
    assert not m.runs and reg.names("twitch") == ["newstreamer"]
