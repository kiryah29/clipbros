"""The /clips review page with reasons, blind review and the feedback API.

ControlPages (control.py) inherits these, so control.py itself only needs a few lines:
    class ControlPages(ReviewPages): ...
and its old clips() and mark() methods removed. Uses self.runs (RunManager, with a
.feedback FeedbackStore), self.posting and self.page from ControlPages.
"""
from __future__ import annotations

import html
import time
from pathlib import Path

from aiohttp import web

from .feedback import STATUS_OF, clip_uid, parse_review, review_bar, review_form, stable_order

e = html.escape


class ReviewPages:
    runs: object
    posting: bool

    async def clips(self, request: web.Request) -> web.Response:
        only = request.query.get("run")
        blind = request.query.get("blind", "1") != "0"
        toggle = "/clips?" + (f"run={only}&" if only else "") + f"blind={'0' if blind else '1'}"
        runs = sorted(self.runs.runs.values(), key=lambda r: r["created"], reverse=True)
        out = [review_bar(blind, toggle)]
        for r in runs:
            if only and r["id"] != only:
                continue
            clips = [c for c in self.runs.clips(r["id"]) if c.reel_path]  # already best score first
            if not clips:
                continue
            rank = {c.id: i for i, c in enumerate(clips, 1)}
            if blind:  # no score order to lean on: same shuffle on every page load
                clips.sort(key=lambda c: stable_order(clip_uid(c.channel, r["id"], c.id)))
            when = time.strftime("%d %b %H:%M", time.localtime(r["started"] or r["created"]))
            out.append(f"<h2>{e(r['channel'])} <span class='mut'>{when} &middot; {e(r['status'])}</span></h2>")
            for c in clips:
                out.append(self._clip_card(r, c, rank[c.id], blind))
        body = "".join(out) if len(out) > 1 else "<p class='mut'>No finished clips yet.</p>"
        return self.page("Clips", "<p><a href='/control'>&larr; Control</a></p>" + body)

    def _clip_card(self, r: dict, c, rank: int, blind: bool) -> str:
        url = f"/runs/{r['id']}/clip/{c.id}"
        uid = clip_uid(c.channel, r["id"], c.id)
        rc = c.receipt
        last = self.runs.feedback.latest(uid)
        # blind: hide everything that hints at the score until someone has decided
        reveal = not blind or last is not None or c.status in ("approved", "rejected", "rejectd", "posted")
        sig = ", ".join(f"{e(k)} {v:.2f}" for k, v in (rc.get("signals") or {}).items()
                        if isinstance(v, (int, float)))
        nd = rc.get("nudity") or {}
        safety = (f"<br><span class='err'>HELD, check before posting: {e(str(nd.get('why', '')))}</span>"
                  if c.status == "held" else
                  f"<br><span class='mut'>nudity check: {e(str(nd.get('why', '')))}</span>"
                  if nd.get("ranges") else "")
        info = (f" &middot; score {c.score:.1f} &middot; <b>{e(c.status)}</b><br>"
                f"<span class='mut'>{e(str(rc.get('why', '')))}</span><br>"
                f"<span class='mut'>triggers: {e(', '.join(rc.get('triggers', [])))} &middot; "
                f"hook: {e(str(rc.get('hook', '')))}</span><br><span class='mut'>signals: {sig}</span>"
                if reveal else
                f" &middot; <b>{e(c.status)}</b><br><span class='mut'>hook: {e(str(rc.get('hook', '')))}</span>")
        posted_ig = (rc.get("posted_to") or {}).get("instagram", "")
        actions = (f"<div class='row'><a class='btn ghost' href='{url}.mp4?dl=1'>Download</a>"
                   + (f"<a class='btn' href='{url}/tiktok'>Post to TikTok</a>"
                      if self.posting and c.status not in ("rejected", "rejectd", "posted") else "")
                   + (f"<a class='btn' href='{url}/instagram'>Post to Instagram</a>"
                      if self.posting and c.status not in ("rejected", "rejectd")
                      and "instagram" not in (rc.get("posted_to") or {}) else "")
                   + (f"<a class='btn ghost' href='{e(posted_ig)}' target='_blank' rel='noopener'>On Instagram</a>"
                      if posted_ig.startswith("https://") else "")
                   + "</div>")
        return (f"<div class='card' id='clip-{c.id}'>"
                f"<video controls playsinline preload='none' data-uid='{e(uid)}' src='{url}.mp4'></video>"
                f"<p><b>Clip #{c.id}</b>{info}{safety}</p>{actions}"
                f"{review_form(url, uid, rank, blind, last)}</div>")

    def _review(self, rid: str, cid: int, action: str, data) -> int:
        c = self.runs.clip(rid, cid)
        if not c:
            raise web.HTTPNotFound()
        review = parse_review(action, data)  # ValueError on bad input
        receipt = Path(c.raw_path).with_name(f"{cid:05d}_receipt.json") if c.raw_path else None
        row = self.runs.feedback.record(clip_uid(c.channel, rid, cid), review, c, run_id=rid,
                                        game=self.runs.runs[rid].get("game") or None,
                                        receipt_path=str(receipt) if receipt else None)
        if c.status != "posted":  # a posted clip stays posted; the review is still saved
            self.runs.set_status(rid, cid, STATUS_OF[action])
        return row

    async def mark(self, request: web.Request) -> web.Response:
        rid = self._rid(request)
        cid = int(request.match_info["cid"])
        try:
            self._review(rid, cid, request.match_info["action"], await request.post())
        except ValueError as ex:
            return self.page("Not saved", f"<p class='err'>{e(str(ex))}</p>"
                             f"<a href='/clips?run={rid}'>Back</a>", status=400)
        raise web.HTTPFound(f"/clips?run={rid}#clip-{cid}")

    async def review_api(self, request: web.Request) -> web.Response:
        """POST /api/runs/{rid}/clip/{cid}/review
        {"decision": "approve"|"reject", "star": bool, "reasons": [...], "note": str,
         "reviewer": str, "watch_ms": int, "rank": int, "blind": bool}"""
        rid = self._rid(request)
        cid = int(request.match_info["cid"])
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Send a JSON object.")
            row = self._review(rid, cid, str(body.get("decision", "")), body)
        except ValueError as ex:
            return web.json_response({"ok": False, "error": str(ex)}, status=400)
        return web.json_response({"ok": True, "id": row})

    async def feedback_export(self, request: web.Request) -> web.Response:
        """GET /api/feedback/export.csv[?days=30]: the newest review per clip and reviewer."""
        try:
            days = float(request.query.get("days", "0"))
        except ValueError:
            days = 0
        since = time.time() - days * 86400 if days > 0 else None
        return web.Response(text=self.runs.feedback.export_csv(since), content_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="clipbros-feedback.csv"'})

    def review_routes(self) -> list:
        return [
            web.post("/api/runs/{rid}/clip/{cid:\\d+}/review", self.review_api),
            web.get("/api/feedback/export.csv", self.feedback_export),
        ]
