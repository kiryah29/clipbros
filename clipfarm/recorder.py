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
import sys
import time
from pathlib import Path

SEG_SECONDS = 2
_JOB = None   # Windows job object holding this process and every child it starts


def _tie_children_to_us() -> None:
    """Make sure streamlink/ffmpeg die with us, even on a hard kill or a closed window.

    Windows: put this process in a job object with KILL_ON_JOB_CLOSE. Children inherit
    the job (including the python.exe that the streamlink.exe launcher starts), and when
    our process ends for any reason Windows closes the job and kills all of them.
    Linux: each child gets PDEATHSIG (see _child_setup). macOS: atexit/stop() only.
    """
    global _JOB
    if _JOB is not None or sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in "abcdef"]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class EXT(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        info = EXT()
        info.Basic.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
                and k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())):
            _JOB = job   # keep the handle open for our whole lifetime
    except Exception as e:  # never block recording over this; stop()/atexit still run
        print(f"  ! could not tie child processes to clipfarm: {e}")


def _child_setup():
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal
            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
        except Exception:
            pass


def _kill_tree(p: subprocess.Popen | None) -> None:
    if not p or p.poll() is not None:
        return
    if sys.platform == "win32":   # the streamlink.exe launcher has a python.exe child
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=5)


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
        _tie_children_to_us()
        extra = {} if sys.platform == "win32" else {"preexec_fn": _child_setup}
        self.sl = subprocess.Popen(["streamlink", "--stdout", self.url, self.quality],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **extra)
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
            stdin=self.sl.stdout, **extra)
        self.sl.stdout.close()   # only ffmpeg reads it now; if ffmpeg dies streamlink sees EOF
        atexit.register(self.stop)  # never leave streamlink/ffmpeg running after we exit

    def stop(self) -> None:
        """Stop streamlink and ffmpeg (and streamlink's child process on Windows)."""
        for p in (self.sl, self.proc):   # streamlink first, so ffmpeg flushes the last segment
            _kill_tree(p)

    @property
    def running(self) -> bool:
        return any(p is not None and p.poll() is None for p in (self.sl, self.proc))

    def segments(self) -> list[tuple[Path, float, float]]:
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
        for f, _, end in self.segments():
            if end < cutoff:
                f.unlink(missing_ok=True)

    def cut(self, start_ts: float, end_ts: float, out: Path) -> float:
        """Copy the buffered segments covering [start_ts, end_ts] into `out` without
        re-encoding (the reel render re-encodes anyway). Segments start on a keyframe,
        so the copy starts at the segment boundary at or before start_ts. Returns that
        lead-in in seconds: the moment starts `lead` seconds into `out`, and craft_clip
        trims it off frame-accurately while rendering."""
        picked = [(f, a) for f, a, b in self.segments() if b > start_ts and a < end_ts]
        if not picked:
            raise RuntimeError("no buffered video for that window (was the recorder running?)")
        first = picked[0][1]
        lead = max(0.0, start_ts - first)
        picked = [f for f, _ in picked]
        listfile = out.with_suffix(".txt")
        listfile.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in picked))
        out.parent.mkdir(parents=True, exist_ok=True)
        _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
              "-i", str(listfile), "-t", f"{lead + end_ts - start_ts:.3f}",
              "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
              "-avoid_negative_ts", "make_zero", str(out)])
        listfile.unlink(missing_ok=True)
        return round(lead, 3)


class DemoRecorder:
    """Generates a 16:9 test clip with a fake 'facecam' box in the configured spot."""

    def __init__(self, facecam: dict, size: str = "1280x720"):
        self.fc = facecam
        self.size = size

    def cut(self, start_ts: float, end_ts: float, out: Path) -> float:
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
        return 0.0   # generated exactly at the window, no lead-in
