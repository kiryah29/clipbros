# clipsbros: live clipping starter

Watches a live stream's **chat**, learns each channel's normal pace, and cuts a clip the
moment chat velocity spikes. Every clip lands in a ranked queue with a **receipt** (the
chat data that explains why it was picked), then gets turned into a 9:16 reel with a
facecam band, word-level captions, a hook and a CTA.

```
live channel ──► chat source ──► spike detector ──► rolling video buffer ──► clip queue ──► reel renderer
 (twitch/yt/kick)   msgs/sec      baseline + z-score    streamlink + ffmpeg     SQLite + receipt   ffmpeg + ASS captions
```

## Try it in 1 minute (no API keys)

Needs Python 3.10+ and `ffmpeg` on your PATH.

```bash
pip install -r requirements.txt
python -m clipfarm demo            # 20 min of simulated chat, detects spikes, renders reels
python -m clipfarm queue           # ranked queue
python -m clipfarm receipt 1       # why clip 1 was picked
python -m pytest -q                # detector, queue and scheduler tests
```

Output goes to `data/clips/<channel>/`: `NNNNN_raw.mp4` (16:9 cut), `NNNNN_reel.mp4`
(1080x1920) and `NNNNN_receipt.json`. Demo footage is a test pattern with a blue box
standing in for the facecam. Use `--no-render` to only run detection.

## Going live

Quick chat-only test (no recording, no keys): `python -m clipfarm watch twitch <channel>`.
It prints the chat pace every 10 s and any spike it would have clipped.


1. `cp .env.example .env` and fill in what you need (Twitch chat works anonymously;
   YouTube needs `YOUTUBE_API_KEY`).
2. In `config/channels.yaml`, enable a channel, set `platform`, and set the `facecam`
   box (fractions of the source frame where the streamer's camera sits).
3. `python -m clipfarm run`. Start it **before** the stream goes live, or add a `schedule`
   so it only watches during the streamer's usual hours.

The live engine only works on live streams. It has to be running while the moment
happens, because the clip is cut from a rolling buffer (`clip.buffer_minutes`) recorded
alongside the chat. It does not process VODs.

## How detection works (`clipfarm/detector.py`)

* Chat is counted in 1 s buckets and smoothed over 3 buckets.
* An EWMA mean + variance (half-life `baseline_halflife_s`) is the channel's "normal
  pace". The std never drops below the Poisson noise floor, so quiet chats don't fire on
  random blips.
* A spike opens when the rate is `z_threshold` std-devs above normal **and** `min_ratio`
  times normal **and** above `min_msgs_per_s`, and stays that way for `min_sustain_s`.
* The baseline is frozen during a spike, then there is a `cooldown_s`.
* The clip window is `peak - pre_roll_s` to `end + post_roll_s`, capped at
  `max_length_s`. Chat reacts a few seconds *after* the moment, which is why pre-roll
  is longer than post-roll.
* Score = weighted z-score, ratio, peak rate and share of hype tokens (KEKW, LUL, CLIP,
  💀...). Weights are in `ranking:`.

On simulated data across 5 seeds (2.5 h of chat, 46 injected bursts) it caught all 46
with 1 false positive. Real chats are messier, so tune per channel in `channels.yaml`.

## Reel craft layer (`clipfarm/craft/`)

| Setting (`reel:`)           | What it does |
|-----------------------------|--------------|
| `layout`                    | `facecam_top` (facecam band / gameplay / optional backdrop) or `gameplay_only` (blurred fill) |
| `facecam_band_ratio`        | Height share of the facecam band |
| `backdrop`                  | Looping filler video under the gameplay (footage you own or licensed) |
| `caption_style`             | `pop` (active word coloured and scaled), `boxed`, `clean` |
| `transcriber`               | `none` or `faster_whisper` (`pip install faster-whisper`) for real word timings |
| `hook_text` / `cta_text`    | Burned-in opener and end card |
| `music` / `music_volume`    | Optional licensed track mixed under the clip audio |

Re-render any clip after changing settings: `python -m clipfarm render <id>`.

## Review workflow

`queue` → watch the reel → `approve <id>` or `reject <id>`. Publishing is deliberately
left manual. Plug in an uploader when you're ready.

## Project layout

```
config/settings.yaml     detector, clip, ranking, reel settings
config/channels.yaml     channels, facecam boxes, schedules, per-channel overrides
clipfarm/chat/           simulated, twitch (IRC/WebSocket), youtube (Data API), kick (stub)
clipfarm/detector.py     chat-velocity spike detection + scoring
clipfarm/recorder.py     rolling live buffer + cutting; demo video generator
clipfarm/queue.py        SQLite ranked queue with receipts
clipfarm/craft/          9:16 reel renderer + ASS captions/hook/CTA
clipfarm/scheduler.py    smart scheduling windows
clipfarm/engine.py       wires it together (demo + live)
```

## Extending

* **Kick**: implement `KickChat.messages()` against Kick's official developer API.
* **Better signal**: add audio loudness or a "clip"-command counter as extra score terms
  in `score_spike`.
* **Uploads**: add a publisher that takes `approved` clips from the queue.

## Rules of the road

* Use official APIs (Twitch IRC/EventSub, YouTube Data API, Kick developer API) and stay
  within their rate limits and quotas. YouTube live chat polling uses quota on every call.
* **Only clip and repost streamers who allow it.** Get permission (many streamers run
  official clipping programs) and credit them in the post.
* Backdrop footage and music must be yours or properly licensed. Gameplay from other
  games, popular songs and "satisfying" videos are common reasons for takedowns and strikes.
* Follow each platform's rules on reuploaded and unoriginal content.

## Post to TikTok (official Content Posting API)

`python -m clipfarm tiktok-app` starts a small web app: connect your TikTok account
(Login Kit), pick a rendered clip, set caption and privacy, tick the consent box and post.
It uses the official API with `FILE_UPLOAD`, so no public video URL is needed.

1. In developers.tiktok.com add **Login Kit** (`user.info.basic`) and **Content Posting API**
   (`video.publish`) to your app and add your own TikTok account as a sandbox target user.
2. TikTok needs an **HTTPS redirect URI** registered on the app. Run a tunnel
   (e.g. `cloudflared tunnel --url http://localhost:8000`) and register
   `https://<tunnel-host>/callback`.
3. Put `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET` and `TIKTOK_REDIRECT_URI` in `.env`
   (see `.env.example`).
4. `pip install -r requirements.txt`, run `python -m clipfarm demo` to get a reel, then
   `python -m clipfarm tiktok-app` and open the tunnel URL in a browser.

Until TikTok audits the app, posts can only be **Only me** (private). The review screen
follows TikTok's UX rules: account shown, no default privacy choice, interaction toggles
off, commercial-content disclosure, consent declaration, and an explicit Post click.
The token is stored in `data/tiktok_token.json` (gitignored).
