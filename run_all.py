"""Keeps the door counters and the dashboard running. Started at Windows logon
by the "Door counter" scheduled task (pythonw, so no console window).

Runs one counter per camera in settings.json and dashboard.py, restarts any
that stop (after 10 s, backing off to 5 min while it keeps failing), and
writes yesterday's daily report just after midnight. Each program's output
goes to its own log (counter_chNN.log, dashboard.log) with RTSP logins masked;
this script logs to supervisor.log.

  pythonw run_all.py      what the scheduled task runs
  python run_all.py       same, in a console (Ctrl+C stops everything)
"""
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta

from common import HERE, load_settings

PYTHON = HERE / ".venv" / "Scripts" / "python.exe"
MASK = re.compile(r"rtsp://[^\s@/]*@")
LOCK_PORT = 8789  # only one supervisor at a time: a second one can't bind it
REPORT_AFTER = 5  # minutes past midnight to write yesterday's report


def log(msg):
    with open(HERE / "supervisor.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def die_together():
    """Put this process (and so every program it starts) in a Windows job that
    is killed as a whole when the supervisor exits for any reason. Otherwise a
    killed supervisor could leave a counter running on its own, and the next
    start would add a second one counting everyone twice."""
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = (k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # extended limits
          and k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
    if not ok:
        log(f"could not tie the programs to the supervisor (error {ctypes.get_last_error()})")
    return job  # the handle must stay open while the supervisor runs


class Program:
    def __init__(self, name, args, log_name):
        self.name, self.args, self.log_path = name, args, HERE / log_name
        self.proc = None
        self.started = 0.0
        self.backoff = 10
        self.retry_at = 0.0

    def start(self):
        self.proc = subprocess.Popen(
            [str(PYTHON), "-u", "-W", "ignore", *self.args], cwd=HERE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW)
        self.started = time.time()
        threading.Thread(target=self._copy_output, args=(self.proc,), daemon=True).start()
        log(f"started {self.name} (pid {self.proc.pid})")

    def _copy_output(self, proc):
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(MASK.sub("rtsp://***@", line))
                f.flush()

    def check(self, now):
        """Restart the program if it has stopped and its wait is over."""
        if self.proc and self.proc.poll() is None:
            return
        if self.proc:
            ran = now - self.started
            self.backoff = 10 if ran > 600 else min(self.backoff * 2, 300)
            log(f"{self.name} stopped (exit {self.proc.returncode} after {ran:.0f} s); restarting in {self.backoff} s")
            self.proc, self.retry_at = None, now + self.backoff
        if now >= self.retry_at:
            self.start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def main():
    lock = socket.socket()
    try:
        lock.bind(("127.0.0.1", LOCK_PORT))
    except OSError:
        log("another supervisor is already running; exiting")
        return
    job = die_together()  # noqa: F841 (kept open on purpose)
    settings = load_settings()
    programs = [Program(f"counter ch{c['channel']}", ["door_counter.py", "--channel", str(c["channel"])],
                        f"counter_ch{c['channel']:02d}.log") for c in settings["cameras"]]
    programs.append(Program("dashboard", ["dashboard.py"], "dashboard.log"))
    log(f"supervisor started: {', '.join(p.name for p in programs)}")
    reported = date.today()
    try:
        while True:
            now = time.time()
            for p in programs:
                p.check(now)
            clock = datetime.now()
            if clock.date() != reported and clock.hour * 60 + clock.minute >= REPORT_AFTER:
                reported = clock.date()
                day = (reported - timedelta(days=1)).isoformat()
                result = subprocess.run([str(PYTHON), "daily_report.py", "--date", day], cwd=HERE,
                                        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                log(f"daily report for {day}: " + (result.stdout.strip().splitlines() or ["no output"])[-1]
                    if result.returncode == 0 else f"daily report for {day} failed: {result.stderr.strip()[-300:]}")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in programs:
            p.stop()
        log("supervisor stopped")


if __name__ == "__main__":
    sys.exit(main())
