"""Measure the untouched organizer command and audit target-free replay."""
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import subprocess
import sys
import time

SUBMISSION = Path("/submission")
DATA = Path("/data")
OUTPUT = Path("/output")
LIMIT = 1800


def timed_command(command):
    started = time.monotonic()
    subprocess.run(command, cwd=SUBMISSION, check=True, timeout=LIMIT)
    return {
        "command": command,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
    }


def audited_run(command, split, stream):
    allowed_data = {DATA / split / name for name in ("queries.csv", "history.csv")}
    used_data = set()
    seen = set()

    def record(kind, **fields):
        print(json.dumps({"event": kind, **fields}, sort_keys=True), file=stream, flush=True)

    def audit(kind, values):
        if kind == "open" and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).resolve()
            mode, flags = values[1:]
            reads = flags & os.O_ACCMODE != os.O_WRONLY
            writes = flags & os.O_ACCMODE != os.O_RDONLY
            if path.is_relative_to(DATA):
                allowed = path in allowed_data and not writes
                if allowed and reads:
                    used_data.add(path)
            elif path.is_relative_to(SUBMISSION):
                allowed = not writes and path.suffix.lower() != ".csv"
            elif path.is_relative_to(OUTPUT):
                allowed = not reads
            else:
                allowed = path.suffix.lower() != ".csv"
            if (path.is_relative_to(DATA) or path.is_relative_to(SUBMISSION)
                    or path.is_relative_to(OUTPUT) or not allowed):
                key = (path, mode, flags)
                if key not in seen:
                    seen.add(key)
                    record("open", path=str(path), mode=mode, flags=flags, allowed=allowed)
            if not allowed:
                raise PermissionError(f"Invalid inference file access: {path}")
        elif kind.startswith("socket.") or kind in {
            "subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"
        }:
            record(kind, allowed=False)
            raise PermissionError(f"Invalid inference operation: {kind}")

    record("start", command=command, split=split)
    sys.addaudithook(audit)
    sys.path.insert(0, str(SUBMISSION))
    sys.argv = command
    runpy.run_path(command[0], run_name="__main__")
    if used_data != allowed_data:
        raise ValueError(f"Observed unexpected data inputs: {used_data}")
    record("end", data_inputs=sorted(map(str, used_data)))


def main():
    mode = sys.argv[1]
    config = json.loads((SUBMISSION / "reproduce.json").read_text())
    command = config["command"]
    if mode == "audit":
        split = sys.argv[2]
        command = command[command.index("predict.py"):]
        command[command.index("--out") + 1] = str(OUTPUT / f"{split}_audited.csv")
        command += ["--split", split]
        with (OUTPUT / f"{split}_trace.jsonl").open("x", buffering=1) as trace:
            audited_run(command, split, trace)
        return

    mountinfo = Path("/proc/self/mountinfo").read_text().splitlines()
    options = {}
    for line in mountinfo:
        fields = line.split()
        if fields[4] in ("/submission", "/data", "/output", "/tmp"):
            options[fields[4]] = fields[5]
    limits = {name: (Path("/sys/fs/cgroup") / name).read_text().strip()
              for name in ("cpu.max", "memory.max")}
    if not all("ro" in options[path].split(",") for path in ("/submission", "/data")):
        raise ValueError(f"Expected read-only data and code: {options}")
    quota, period = map(int, limits["cpu.max"].split())
    assert quota == 2 * period
    assert int(limits["memory.max"]) == 16 * 1024 ** 3
    test = timed_command(command)
    for split in ("test", "validation"):
        timed_command([sys.executable, "-I", "-B", __file__, "audit", split])
    report = {"python": sys.version, "platform": platform.platform(), "limits": limits,
              "timeout_seconds_per_command": LIMIT, "mount_options": options, "test": test}
    (OUTPUT / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
