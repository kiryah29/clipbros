from .base import ChatSource


def make_chat_source(channel: dict) -> ChatSource:
    p = channel["platform"]
    if p == "simulated":
        from .simulated import SimulatedChat
        return SimulatedChat(channel["name"], realtime=True, **channel.get("sim", {}))
    if p == "twitch":
        from .twitch import TwitchChat
        return TwitchChat(channel["name"])
    if p == "youtube":
        from .youtube import YouTubeChat
        return YouTubeChat(channel["name"], video_id=channel.get("video_id") or None)
    if p == "kick":
        from .kick import KickChat
        return KickChat(channel["name"])
    raise ValueError(f"unknown platform {p!r}")
