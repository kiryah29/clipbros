# clipsbros: live clipping starter

Watches a live stream's **chat, audio and speech**, learns each streamer's normal level for
each, and cuts a clip the moment chat or loudness spikes. Every clip gets a 0-100 score from
six weighted signals and lands in a ranked queue with a **receipt** (the data that explains
why it was picked), then gets turned into a 9:16 reel with a facecam band, word-level
captions, a hook and a CTA.

```
live channel ─┬► chat source ──► chat spike detector ─┐
              └► rolling video buffer ─┬► loudness ───┼► clip window ──► score ──► clip queue ──► reel renderer
                 streamlink + ffmpeg   └► speech-to-text (pace + phrases, captions)
```

## Try it in 1 minute (no API keys)

Needs Python 3.10+ and `ffmpeg` on your PATH.

```bash
pip install -r requirements.txt
python -m clipfarm demo            # 20 min of simulated chat, detects spikes, renders reels
python -m clipfarm queue           # ranked queue
python -m clipfarm receipt 1       # why clip 1 was picked
python -m pytest -q                # detector, signals, queue and scheduler tests
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
   box (fractions of the source frame where the streamer's camera sits). Leave it out
   (or run a channel that isn't listed, e.g. from the control page) and each clip is
   searched for a face that stays put; the webcam overlay around it becomes the top band.
   No face found means a gameplay-only reel. The receipt says which (`facecam_source`).
3. Speech-to-text (once): `python -m clipfarm setup-speech` downloads a small model
   (~100 MB, into `models/`). Without it, speech scores 0 and everything else still works.
   Nudity check (once, optional): `python -m clipfarm setup-nudity` fetches its model
   (~12 MB). It is also fetched on the first clip; see "Nudity check" below.
4. `python -m clipfarm run`. Start it **before** the stream goes live, or add a `schedule`
   so it only watches during the streamer's usual hours.

The live engine only works on live streams. It has to be running while the moment
happens, because the clip is cut from a rolling buffer (`clip.buffer_minutes`) recorded
alongside the chat. It does not process VODs.

One-off test on any channel without editing `channels.yaml` (records and renders, never
posts): `python -m clipfarm run twitch <channel> --game gta6`. Ctrl+C stops it and stops
streamlink/ffmpeg with it.

## Moment score (`clipfarm/signals.py`)

Each signal is compared with the streamer's own normal level: 0 points at normal, half
points at 3x normal (`signals.trigger_ratio`), full points at 5x. Score = sum of
weight x strength, 0-100. Weights live in `signals.weights` and must add up to 1.0:

| Signal         | Weight | What it measures |
|----------------|-------:|------------------|
| `audio`        | 0.35 | Stream loudness vs normal (`clipfarm/audio.py`). 3x normal (about +10 dB) for 1 s also **opens a clip** on its own |
| `speech`       | 0.20 | Streamer speech (`clipfarm/speech.py`): half talking pace vs normal, half excited phrases ("no way", "oh my god", "no no no"; list in `speech.keywords`) |
| `chat_speed`   | 0.22 | Chat messages/s vs normal (the chat spike detector below) |
| `chat_words`   | 0.13 | Share of hype words in chat (`ranking.hype_tokens`); half the messages = full points |
| `game_events`  | 0.07 | Game-profile moment match (`config/games/`) |
| `twitch_clips` | 0.03 | Viewers clipping it on Twitch. Not wired up yet, scores 0 |

A scream and the chat reaction a few seconds later become **one** clip: triggers that
overlap a clip still waiting to be cut are merged into it. The receipt shows every signal's
points (`score_points`), the loudness ratio, the phrases heard and the transcript.

**Audio**: every finished 2 s buffer segment is decoded once (16 kHz mono). Each second's
value is its loudest 250 ms RMS, so a short yell isn't averaged away. Settings in `audio:`
(`min_ratio`, `warmup_s`, `pre_roll_s` 15 s since a scream is AT the moment).

**Speech**: the same audio is transcribed in 4 s chunks while the stream runs. Engines
(`speech.engine`):

| Engine | Install | Notes |
|--------|---------|-------|
| `sherpa_onnx` | in requirements.txt, then `python -m clipfarm setup-speech` | Works on Windows ARM64. Word timings are spread over each 4 s chunk |
| `faster_whisper` | `pip install faster-whisper` | Exact word timings. x64 PCs only (no Windows ARM64 build) |
| `auto` (default) | | faster-whisper if installed, else sherpa-onnx, else off |

The transcript of the clip window is saved in the receipt and used for the reel captions
(`reel.transcriber: auto`). If the PC can't keep up it skips old chunks and says so.

## ClipAgent check (`clipfarm/agent.py`)

After a clip window is scored, ClipAgent decides whether it is really clip-worthy, labels
it, and explains why. A keyword is evidence, never a trigger: one `Pog` in a busy chat
counts for nothing, a bucket only counts when at least 10% of the spike's messages hit it.

Six checks: chat keywords/emotes, chat speed spike, intense chat (30% of messages with
`!`/`?`, or one emote in 30% of messages), audio spike (3x normal), a spoken trigger
phrase, and a context event (game-profile moment, donation read, streamer reading chat,
rage quit). A clip needs at least 2 checks, one of them a chat or audio spike.

- **clip_type**: hype, funny, clutch, fail, rage, donation, chat_interaction or other,
  voted on by the keyword buckets, phrases, context and game moment.
- **confidence** 0..1: how many checks agree plus how strong the moment score is, minus
  0.1 per conflict (chat laughing while the streamer rages; loud stream, quiet chat).
- Clips under `clip_agent.min_confidence` (0.65) are skipped (`gate: false` keeps them).
- Each clip gets `NNNNN_candidate.json` in the spec's JSON format next to its receipt.
  `python -m clipfarm candidates` lists them ranked by confidence (`--all` adds the low ones).
- Keyword lists live in `settings.yaml` under `clip_agent`. A game profile or channel can
  replace a bucket, or add to it with `extra_chat_keywords` / `extra_speech_phrases`
  (the GTA 6 profile adds LMAO, 💀, WASTED...).
- The clip type also steers the hook card (`type_clutch`, `type_donation`, ... in hooks.py).

## How chat detection works (`clipfarm/detector.py`)

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
* The spike's ratio and hype-word share feed the `chat_speed` and `chat_words` signals.

On simulated data across 5 seeds (2.5 h of chat, 46 injected bursts) it caught all 46
with 1 false positive. Real chats are messier, so tune per channel in `channels.yaml`.

## Reel craft layer (`clipfarm/craft/`)

| Setting (`reel:`)           | What it does |
|-----------------------------|--------------|
| `layout`                    | `facecam_top` (facecam band / gameplay / optional backdrop) or `gameplay_only` (blurred fill) |
| `facecam_band_ratio`        | Height share of the facecam band |
| `facecam_auto`              | Find the facecam in the clip when the channel has no `facecam` box (needs `opencv-python-headless`; model in `assets/models`) |
| `backdrop`                  | Looping filler video under the gameplay (footage you own or licensed) |
| `caption_style`             | `clipbros` (bold white caps, spoken word in lime), `pop`, `boxed`, `clean` |
| `transcriber`               | `auto` (the live stream transcript, see Speech above), `faster_whisper`, or `none` |
| `hooks`, `hook_text` / `hook_seconds` | Hook card on the facecam/gameplay seam (`0` = whole clip). Text is written per clip from what happened (`clipfarm/hooks.py`): moment type, what the streamer said, what chat spammed, loudness, chat speed; no repeats across the last 6 clips. `hook_text` is the fallback |
| `cta_text`                  | End card (empty = none; a hard cut loops better) |
| `accent_color`, `hook_bg`, `hook_color`, `show_logo`, `logo_height`, `logo` | ClipBros brand layer (`craft/overlay.py`, tokens in `brand.py`) |
| `caption_words`, `caption_outline` | words on screen at once, Ink outline px |
| `safe_zone`                 | Pixels kept clear for TikTok's side buttons (right) and description/sound bar (bottom) |
| `music` / `music_volume`    | Optional licensed track mixed under the clip audio |

Re-render any clip after changing settings: `python -m clipfarm render <id>`.

The default look follows the ClipBros Brand Kit (on-clip page): facecam top 30 %, a Clip Lime
line on the seam, the hook in Inter 800 Ink on a Paper card straddling it, the primary lockup
top-left with one bracket length of clear space, and captions in Inter 800 caps (Paper with an
Ink outline, four words at a time) with the spoken word in Clip Lime, everything inside the
TikTok safe zone. Fonts (Inter ExtraBold and Space Grotesk Bold, both OFL) ship in
`assets/fonts/` so reels look the same on every PC.

### Brand kit in code (`clipfarm/brand.py`)

Palette, fonts and logo rules live in one module: Ink `#131119`, Night `#1D1A26`, Paper
`#F3F0E8`, Clip Lime `#C8F03C`, Stream Violet `#7C5CFF`, Fog `#A49FB2`. The logo (corner
brackets + play button, two-tone "ClipBros" wordmark) is drawn from geometry, so it stays sharp
at any size, and falls back to the mark alone below the kit's 120 px minimum. The review web
app and the `docs/` pages use the same tokens (Ink theme with Lime, Paper theme with Violet).
Regenerate the logo PNGs (mark, lockups on Ink and on Paper, TikTok profile picture) with
`python -m clipfarm.brand`.

## Game profiles (`config/games/`)

Set `game: gta6` (and `streamer: "Name"` for hooks) on a channel to load
`config/games/gta6.yaml`. A profile adds:

* **moments**: kinds of moment (cops/chase, new mechanic, chaos/fail, hype let-down,
  graphic shock) with the chat words viewers type when they happen. A spike gets the best
  matching label in its receipt and scores the `game_events` signal.
* **clip windows per moment**: setup-heavy moments (a chase) get a 45 s pre-roll; quick
  reactions 15 s. Post-roll is short so the clip hard-cuts after the reaction and loops.
* **hooks per moment**, e.g. "DO NOT DO THIS IN GTA 6". `{streamer}` and `{game}` fill in.
* **safety holds**: moments marked `safety: review` (gore) set `safety_review: true` in the
  receipt so they are never auto-posted.

Merge order is `settings.yaml` < game profile < the channel's own overrides.

## Nudity check

Every clip is scanned before its reel is rendered (`clipfarm/nudity.py`, `nudity:` in
`settings.yaml`). Three frames a second go through NudeNet's exposed-body-part detector
(ONNX, runs on onnxruntime or OpenCV, so on the Windows ARM64 PC too), at 320 and 640 px
so a small naked figure in a wide shot is caught as well as a close-up.

* **clear nudity**: those seconds (plus half a second each side) are blurred hard in the
  reel's gameplay. Facecam, hook card and captions stay sharp. The clip is `crafted` as usual.
* **unsure** (a frame outside the blurred part scores just under the blur line), **mostly
  nude** (over half the clip would be blurred) or **check could not run**: the clip is
  rendered but its status is `held`, the control page and TikTok page flag it, and it
  needs a human look before posting.

The result is in the receipt under `nudity` (decision, blurred ranges, labels, why).
Try it on any video: `python -m clipfarm setup-nudity path/to/clip.mp4`.
Model licence: the NudeNet package is published as MIT on PyPI, but its GitHub repo
carries an AGPL-3.0 licence file, so the model is downloaded, not committed.

## Review workflow

`queue` → watch the reel → `approve <id>` or `reject <id>`. Publishing is deliberately
left manual. Plug in an uploader when you're ready.

## Project layout

```
config/settings.yaml     detector, clip, ranking, reel settings
config/channels.yaml     channels, facecam boxes, schedules, per-channel overrides
clipfarm/chat/           simulated, twitch (IRC/WebSocket), youtube (Data API), kick (stub)
clipfarm/detector.py     chat-velocity spike detection
clipfarm/signals.py      per-streamer baselines + the weighted 0-100 moment score
clipfarm/audio.py        stream loudness signal (also opens clips on screams)
clipfarm/speech.py       speech-to-text (sherpa-onnx / faster-whisper), pace + phrases
clipfarm/recorder.py     rolling live buffer + cutting; demo video generator
clipfarm/queue.py        SQLite ranked queue with receipts
config/games/            game profiles (moment hints, hooks, clip windows)
clipfarm/moments.py      labels spikes with a game moment, picks hook + clip window
clipfarm/craft/          9:16 reel renderer, ASS captions, brand overlay (logo/hook card)
clipfarm/scheduler.py    smart scheduling windows
clipfarm/engine.py       wires it together (demo + live)
```

## Extending

* **Kick**: implement `KickChat.messages()` against Kick's official developer API.
* **Twitch clips signal**: count clips created on Twitch (Helix `GET /clips`) in the window
  and fill `twitch_clips` in `engine.score_moment`.
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

## Control page (test runs from another computer)

The `tiktok-app` web app also has a password-protected control page at `/control`.
Someone with the shared password can start a test run on a Twitch channel (1 to 120
minutes, game profile optional), see whether it is live and how many clips it made, stop it,
and watch, download, approve or reject the reels at `/clips`. Nothing is posted.

- At most 2 runs at once; up to 5 more wait in line. Each run has its own folder
  `data/control/<run id>/` (queue.db, clips, `run.log`), so your own queue is untouched.
- Stop kills the run and its streamlink/ffmpeg; runs also die when the app stops.
- Run folders are deleted `CONTROL_KEEP_DAYS` days after they end (default 3).
- Every streamer you start a run for is saved in `data/streamers.json` (first/last seen,
  number of runs, last game) and kept after their run folders are gone. They are listed
  under "Saved streamers" and suggested in the channel box. `clipfarm run` saves the
  channels it monitors there too.
- Public without a password: `/`, terms, privacy, the TikTok verification files and `/callback`.

`.env` settings:

| Setting | What it does |
|---|---|
| `CONTROL_PASSWORD_HASH` | the shared password, hashed. Make it with `python -m clipfarm control-password`. Without it nobody can sign in. |
| `CONTROL_KEEP_DAYS` | days to keep finished runs (default 3) |
| `CONTROL_ALLOW_POSTING` | set to `1` to turn on the TikTok review/post pages (`/tiktok`). Off by default, also for you. |
