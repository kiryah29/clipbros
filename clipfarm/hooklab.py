"""Hook lab: the hook on each clip next to the rule writer's candidates, before anything posts.

For every finished clip of the control runs it shows the hook burned on the video, what
today's templates would pick now, the rule writer's top candidates and the facts behind
them. You mark the best one (or "none are good") and flag bad ones; picks are saved in
data/hooklab/picks.json, so the summary shows which styles win and how often the rule
writer finds a hook at all. Nothing is posted and no clip is changed.

  web   clipbros.app/hooklab (signed in)
  cli   python -m clipfarm hooklab [--run <run id>]
"""
from __future__ import annotations

import html
import json
import time
from collections import Counter
from pathlib import Path

from aiohttp import web

from .config import channel_settings, load_channels, load_settings
from .queue import ClipQueue
from .hookwriter import hook_cfg, pick_hook

e = html.escape


def run_cfg(channel: str, game: str = "", settings: dict | None = None) -> dict:
    """Settings for a channel as a run would use them: its channels.yaml entry if it has one
    (display name, pronoun, game), with the run's game profile on top."""
    settings = settings or load_settings()
    try:
        entry = next((c for c in load_channels() if c.get("name", "").lower() == channel.lower()), None)
    except (OSError, KeyError, TypeError, AttributeError):
        entry = None
    ch = dict(entry or {"name": channel, "platform": "twitch"})
    if game:
        ch["game"] = game
    try:
        return channel_settings(settings, ch)
    except FileNotFoundError:            # a profile that no longer exists
        ch.pop("game", None)
        return channel_settings(settings, ch)


def reel_view(receipt: dict) -> tuple[list | None, list, float | None, str]:
    """(words, chat, peak, transcript source), all timed from the same zero. Clips crafted
    with the clip transcriber have words timed from the reel start; older clips only have
    the live transcript, timed from the clip window start like the chat log."""
    anchor = receipt.get("anchor") or {}
    shift = float(anchor.get("shift_s") or 0)
    chat = [[float(m[0]) - shift, *m[1:]] for m in receipt.get("chat") or []]
    if receipt.get("caption_transcript"):
        peak = anchor.get("peak_s")
        if peak is None and receipt.get("peak_ts") and receipt.get("start_ts"):
            peak = receipt["peak_ts"] - receipt["start_ts"] - shift
        return receipt["caption_transcript"], chat, peak, "clip"
    peak = (receipt["peak_ts"] - receipt["start_ts"]
            if receipt.get("peak_ts") and receipt.get("start_ts") else None)
    return receipt.get("transcript"), receipt.get("chat") or [], peak, "live"


def review(receipt: dict, cfg: dict) -> dict:
    """Everything the lab shows for one clip."""
    words, chat, peak, source = reel_view(receipt)
    seed = int(receipt.get("peak_ts") or receipt.get("start_ts") or 0)
    rules = pick_hook(receipt, cfg, None, words, chat, peak, seed, writer="rules")
    now = pick_hook(receipt, cfg, None, words, chat, peak, seed, writer="templates")
    return {
        "burned": receipt.get("hook") or "", "burned_kind": receipt.get("hook_kind") or "",
        "templates": now["hook"], "templates_kind": now["hook_kind"],
        "rules": rules["hook"], "rules_kind": rules["hook_kind"], "rules_writer": rules["hook_writer"],
        "candidates": rules.get("hook_candidates") or [], "facts": rules.get("hook_facts") or {},
        "transcript": source,
        "min_score": float(hook_cfg(cfg).get("min_score", 1.0)),
    }


class Picks:
    """Your picks, one per clip: data/hooklab/picks.json."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.items: dict[str, dict] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.items = {}

    def set(self, key: str, entry: dict) -> None:
        self.items[key] = {**entry, "ts": int(time.time())}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def summary(self) -> dict:
        picks = list(self.items.values())
        won = Counter(p.get("style") for p in picks if p.get("choice") == "new")
        bad = Counter(b[1] for p in picks for b in p.get("bad") or [] if len(b) > 1)
        return {"reviewed": len(picks),
                "new": sum(p.get("choice") == "new" for p in picks),
                "old": sum(p.get("choice") == "old" for p in picks),
                "none": sum(p.get("choice") == "none" for p in picks),
                "fallback": sum(p.get("choice") == "fallback" for p in picks),
                "styles_won": dict(won.most_common()), "styles_flagged": dict(bad.most_common())}


def lab_items(runs, only: str | None = None) -> list[tuple[dict, object, dict]]:
    """(run, clip, review) for every finished clip of the control runs, newest run first."""
    settings = load_settings()
    out = []
    for r in sorted(runs.runs.values(), key=lambda r: r["created"], reverse=True):
        if only and r["id"] != only:
            continue
        clips = [c for c in runs.clips(r["id"]) if c.reel_path]
        if not clips:
            continue
        cfg = run_cfg(r["channel"], r.get("game") or "", settings)
        for c in sorted(clips, key=lambda c: c.id):
            out.append((r, c, review(c.receipt, cfg)))
    return out


def coverage(items) -> dict:
    """How often the rule writer found a hook itself instead of falling back."""
    n = len(items)
    found = sum(rv["rules_writer"] == "rules" for _, _, rv in items)
    styles = Counter(rv["rules_kind"] for _, _, rv in items if rv["rules_writer"] == "rules")
    return {"clips": n, "rules": found, "fallback": n - found, "styles": dict(styles.most_common())}


# --------------------------------------------------------------------------- page
class HookLabPages:
    def __init__(self, runs, page):
        self.runs, self.page = runs, page
        self.picks = Picks(runs.root.parent / "hooklab" / "picks.json")

    def _summary_html(self, items) -> str:
        cov, s = coverage(items), self.picks.summary()
        pct = lambda a, b: f"{round(100 * a / b)}%" if b else "-"  # noqa: E731
        styles = ", ".join(f"{e(k)} {v}" for k, v in cov["styles"].items()) or "none"
        won = ", ".join(f"{e(str(k))} {v}" for k, v in s["styles_won"].items()) or "-"
        flagged = ", ".join(f"{e(str(k))} {v}" for k, v in s["styles_flagged"].items()) or "-"
        return (f"<div class='card'><b>Rule writer found a hook on {cov['rules']} of {cov['clips']} clips "
                f"({pct(cov['rules'], cov['clips'])})</b>; the rest fall back to templates.<br>"
                f"<span class='mut'>styles: {styles}</span><br>"
                f"<b>Your picks: {s['reviewed']} reviewed</b> &middot; new hook better {s['new']} "
                f"({pct(s['new'], s['reviewed'])}) &middot; fallback better {s['fallback']} &middot; "
                f"old hook better {s['old']} &middot; none good {s['none']}"
                f"<br><span class='mut'>styles picked: {won} &middot; flagged bad: {flagged}</span></div>")

    def _clip_html(self, r: dict, c, rv: dict, only: str | None = None) -> str:
        key = f"{r['id']}/{c.id}"
        mine = self.picks.items.get(key) or {}
        url = f"/runs/{r['id']}/clip/{c.id}"
        opts = [("old", f"On the video: {rv['burned']}", e(rv["burned_kind"]))]
        if rv["rules_writer"] != "rules" and rv["rules"] != rv["burned"]:
            opts.append(("f", rv["rules"], f"fallback ({e(rv['rules_kind'])}): no rule hook fits"))
        for i, (text, style, sc) in enumerate(rv["candidates"]):
            mark = "" if sc >= rv["min_score"] else " (below min score)"
            opts.append((f"c{i}", text, f"{e(style)} &middot; score {sc}{mark}"))
        rows = []
        for val, text, why in opts:
            checked = " checked" if mine.get("pick") == val else ""
            flag = (f"<label class='mut'><input type='checkbox' name='bad' value='{val}'"
                    f"{' checked' if val in (mine.get('bad_keys') or []) else ''}> bad</label>"
                    if val not in ("old", "f") else "")
            rows.append(f"<div class='row'><label><input type='radio' name='pick' value='{val}'{checked}> "
                        f"<b>{e(text)}</b> <span class='mut'>{why}</span></label>{flag}</div>")
        rows.append(f"<div class='row'><label><input type='radio' name='pick' value='none'"
                    f"{' checked' if mine.get('pick') == 'none' else ''}> None of these are good</label></div>")
        facts = ", ".join(f"{e(k)}: {e(str(v))}" for k, v in rv["facts"].items()
                          if k not in ("streamer", "game")) or "none found"
        done = (f" &middot; <span class='ok'>reviewed</span>" if mine else "")
        return (f"<div class='card' id='c-{e(r['id'])}-{c.id}'><video controls playsinline preload='none' "
                f"src='{url}.mp4'></video><p><b>{e(r['channel'])} #{c.id}</b>{done}<br>"
                f"<span class='mut'>rule pick: {e(rv['rules'])} ({e(rv['rules_writer'])}: {e(rv['rules_kind'])}) &middot; "
                f"templates now: {e(rv['templates'])} &middot; transcript: {e(rv['transcript'])}</span></p>"
                f"<form method='post' action='/hooklab/pick'>"
                f"<input type='hidden' name='rid' value='{e(r['id'])}'><input type='hidden' name='cid' value='{c.id}'>"
                f"<input type='hidden' name='view' value='{e(only or '')}'>"
                + "".join(rows) +
                f"<details><summary class='mut'>Facts behind the candidates</summary><p class='mut'>{facts}</p></details>"
                f"<div class='row'><input type='text' name='note' maxlength='200' placeholder='Note (optional)' "
                f"style='flex:1' value='{e(mine.get('note', ''))}'><button>Save</button></div></form></div>")

    async def lab(self, request: web.Request) -> web.Response:
        only = request.query.get("run")
        items = lab_items(self.runs, only)
        body = ["<p><a href='/control'>&larr; Control</a></p><h2>Hook lab</h2>",
                "<p class='mut'>Pick the best hook per clip, or say none are good. Flag a candidate as bad if "
                "it reads wrong or isn't true. Nothing is posted; this only decides whether the new writer "
                "gets switched on (reel.hooks.writer: rules).</p>", self._summary_html(items)]
        last = None
        for r, c, rv in items:
            if r["id"] != last:
                when = time.strftime("%d %b %H:%M", time.localtime(r.get("started") or r["created"]))
                body.append(f"<h2>{e(r['channel'])} <span class='mut'>{when} &middot; "
                            f"{e(r.get('game') or 'default')}</span></h2>")
                last = r["id"]
            body.append(self._clip_html(r, c, rv, only))
        if not items:
            body.append("<p class='mut'>No finished clips in the control runs yet.</p>")
        return self.page("Hook lab", "".join(body))

    async def pick(self, request: web.Request) -> web.Response:
        form = await request.post()
        rid, cid = str(form.get("rid", "")), str(form.get("cid", ""))
        if rid not in self.runs.runs or not cid.isdigit():
            raise web.HTTPNotFound()
        clip = self.runs.clip(rid, int(cid))
        if not clip:
            raise web.HTTPNotFound()
        r = self.runs.runs[rid]
        rv = review(clip.receipt, run_cfg(r["channel"], r.get("game") or ""))
        cands = {f"c{i}": c for i, c in enumerate(rv["candidates"])}
        pick = str(form.get("pick", ""))
        bad_keys = [b for b in form.getall("bad", []) if b in cands]
        entry = {"pick": pick, "bad_keys": bad_keys, "bad": [cands[b][:2] for b in bad_keys],
                 "old": rv["burned"], "candidates": rv["candidates"], "facts": rv["facts"],
                 "channel": r["channel"], "note": str(form.get("note", ""))[:200]}
        if pick in cands:
            entry.update(choice="new", text=cands[pick][0], style=cands[pick][1])
        elif pick == "f":
            entry.update(choice="fallback", text=rv["rules"], style="fallback:" + rv["rules_kind"])
        elif pick in ("old", "none"):
            entry.update(choice=pick)
        else:
            entry.update(choice="skipped")
        self.picks.set(f"{rid}/{cid}", entry)
        back = "?run=" + rid if form.get("view") == rid else ""
        raise web.HTTPFound(f"/hooklab{back}#c-{rid}-{cid}")

    def routes(self) -> list:
        return [web.get("/hooklab", self.lab), web.post("/hooklab/pick", self.pick)]


class ReadOnlyRuns:
    """The control runs as the lab needs them, without RunManager's side effects (it marks
    running runs interrupted and deletes old ones), so the CLI is safe while the app runs."""

    def __init__(self, root: Path):
        self.root, self.runs = root, {}
        for f in root.glob("*/run.json"):
            try:
                r = json.loads(f.read_text())
                self.runs[r["id"]] = r
            except (OSError, ValueError, KeyError):
                continue

    def clips(self, rid: str) -> list:
        db = self.root / rid / "queue.db"
        if not db.is_file():
            return []
        q = ClipQueue(db)
        try:
            return q.ranked(None, 200)
        finally:
            q.db.close()


def print_lab(runs, only: str | None = None) -> None:
    """Terminal version: per clip the hook on the video and the rule writer's candidates."""
    items = lab_items(runs, only)
    for r, c, rv in items:
        print(f"{r['channel']} #{c.id}  [{rv['transcript']} transcript]")
        print(f"   on video : {rv['burned']}")
        print(f"   templates: {rv['templates']}  ({rv['templates_kind']})")
        for text, style, sc in rv["candidates"]:
            print(f"   {sc:>5.2f} {style:<13} {text}")
        if not rv["candidates"]:
            print("   (no rule candidates)")
    cov = coverage(items)
    print(f"\nrule writer found a hook on {cov['rules']} of {cov['clips']} clips; styles: {cov['styles']}")
