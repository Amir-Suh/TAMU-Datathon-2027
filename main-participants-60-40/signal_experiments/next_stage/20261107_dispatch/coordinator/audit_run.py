"""Run one Python worker process with guarded and logged local file access."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("fit-predict", "score"))
    parser.add_argument("--observe-root", action="append", type=Path, default=[])
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    roots = (workspace, *(root.resolve() for root in args.observe_root))
    script = args.script.resolve(strict=True)
    seen = set()

    with args.trace.open("x", encoding="utf-8", buffering=1) as trace:
        def record(event, **fields):
            trace.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")

        def audit(event, values):
            if event == "open":
                name, mode, flags = values
                if not isinstance(name, (str, bytes, os.PathLike)):
                    return
                path = Path(os.fsdecode(name)).resolve()
                parts = {p.lower() for p in path.parts}
                filename = path.name.lower()
                reading = flags & os.O_ACCMODE != os.O_WRONLY
                validation_only = filename in {
                    "targets.csv", "labels.csv", "reference.csv", "conditions.csv"
                }
                test_private = "test" in parts and validation_only
                held_answers = parts.intersection({"validation", "lockbox"}) and validation_only
                allowed = not reading or not (
                    test_private or (args.stage == "fit-predict" and held_answers)
                )
                local = any(path.is_relative_to(root) for root in roots)
                key = (path, mode, flags)
                if (local or not allowed) and key not in seen:
                    seen.add(key)
                    shown = str(path.relative_to(workspace)) if path.is_relative_to(workspace) else str(path)
                    record("open", path=shown, mode=mode, flags=flags, allowed=allowed)
                if not allowed:
                    raise PermissionError(f"Protected data read during {args.stage}: {path}")
            elif event in {"subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"}:
                record("child_process", audit_event=event, command=str(values[0]))
            elif event == "socket.connect":
                record("socket_connect", allowed=False)
                raise PermissionError("Network connections are disabled in audit runs")

        record("start", stage=args.stage, script=str(script),
               args=args.script_args, observed_roots=[str(root) for root in roots],
               utc=datetime.now(timezone.utc).isoformat())
        sys.addaudithook(audit)
        sys.argv = [str(script), *args.script_args]
        sys.path.insert(0, str(script.parent))
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            record("end", exit=exc.code)
            raise
        except BaseException as exc:
            record("end", error=type(exc).__name__, message=str(exc))
            raise
        else:
            record("end", exit=0)


if __name__ == "__main__":
    main()
