"""Video capture.

LiveRecorder keeps a rolling buffer of short .ts segments per channel
(streamlink -> ffmpeg segment muxer), so when a spike closes we can cut a clip
that starts BEFORE the moment chat reacted. DemoRecorder fabricates a test
video so the pipeline runs offline.
"""
from __future__ import annotations

import atexit
import shutil
import subprocess
import time
from pathlib import Path

SEG_SECONDS = 2


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


class LiveRecorder:
    def __init__(self, channel: str, stream_url: str, workdir: Path, quality: str,
                 buffer_minutes: float):
        self.channel = channel
        self.url = stream_url
        self.dir = Path(workdir) / "buffer" / channel
        self.dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self.keep_s = buffer_minutes * 60
        self.proc: subprocess.Popen | None = None
        self.sl: subprocess.Popen | None = None

    def start(self) -> None:
        if not shutil.which("streamlink"):
            raise RuntimeError("streamlink is not installed (pip install streamlink)")
        self.sl = subprocess.Popen(["streamlink", "--stdout", self.url, self.quality],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # segment list (csv: file,start,end relative to t0) works on every OS,
        # unlike strftime("%s") which Windows lacks
        for f in self.dir.glob("seg_*.ts"):
            f.unlink(missing_ok=True)
        self.list_path = self.dir / "segments.csv"
        self.list_path.unlink(missing_ok=True)
        self.t0 = time.time()
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-c", "copy", "-f", "segment",
             "-segment_time", str(SEG_SECONDS), "-reset_timestamps", "1",
             "-segment_list", str(self.list_path), "-segment_list_type", "csv",
             str(self.dir / "seg_%06d.ts")],
            stdin=self.sl.stdout)
        atexit.register(self.stop)  # never leave streamlink/ffmpeg running after we exit

    def stop(self) -> None:
        for p in (self.proc, self.sl):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()

    def _segments(self) -> list[tuple[Path, float, float]]:
        """(file, abs_start, abs_end) for finished segments still on disk."""
        out = []
        if not getattr(self, "list_path", None) or not self.list_path.exists():
            return out
        for line in self.list_path.read_text().splitlines():
            try:
                name, a, b = line.rsplit(",", 2)
            except ValueError:
                continue
            f = self.dir / name
            if f.exists():
                out.append((f, self.t0 + float(a), self.t0 + float(b)))
        return out

    def prune(self) -> None:
        cutoff = time.time() - self.keep_s
        for f, _, end in self._segments():
            if end < cutoff:
                f.unlink(missing_ok=True)

    def cut(self, start_ts: float, end_ts: float, out: Path) -> Path:
        picked = [(f, a) for f, a, b in self._segments() if b > start_ts and a < end_ts]
        if not picked:
            raise RuntimeError("no buffered video for that window (was the recorder running?)")
        first = picked[0][1]
        picked = [f for f, _ in picked]
        listfile = out.with_suffix(".txt")
        listfile.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in picked))
        out.parent.mkdir(parents=True, exist_ok=True)
        _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
              "-i", str(listfile), "-ss", f"{max(0, start_ts - first):.2f}",
              "-t", f"{end_ts - start_ts:.2f}", "-c:v", "libx264", "-preset", "veryfast",
              "-c:a", "aac", str(out)])
        listfile.unlink(missing_ok=True)
        return out


class DemoRecorder:
    """Generates a 16:9 test clip with a fake 'facecam' box in the configured spot."""

    def __init__(self, facecam: dict, size: str = "1280x720"):
        self.fc = facecam
        self.size = size

    def cut(self, start_ts: float, end_ts: float, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        dur = end_ts - start_ts
        fc = self.fc
        box = (f"drawbox=x=iw*{fc['x']}:y=ih*{fc['y']}:w=iw*{fc['w']}:h=ih*{fc['h']}"
               f":color=0x3b82f6@1:t=fill")
        _run(["ffmpeg", "-y", "-loglevel", "error",
              "-f", "lavfi", "-i", f"testsrc2=size={self.size}:rate=30:duration={dur:.2f}",
              "-f", "lavfi", "-i", f"sine=frequency=330:duration={dur:.2f}",
              "-vf", box, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
              "-c:a", "aac", "-shortest", str(out)])
        return out
