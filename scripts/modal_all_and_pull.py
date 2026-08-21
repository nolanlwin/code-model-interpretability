#!/usr/bin/env python3
"""Launch the 6-job class_struct sweep on Modal, then pull whatever landed.

Starts the GPU work with --detach (closing this script does NOT kill Modal).
Polls until the app stops (success, crash, or $30 credit exhaustion) and
syncs /results off the volume into ./results/modal.

Sleep/idle is fine. Closing the lid kills this puller only — the sweep keeps
going. In that case, rerun:

  modal volume get --force class-struct-data /results ./results/modal
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VOLUME = "class-struct-data"
REMOTE = "/results"
LOCAL = REPO / "results" / "modal"
POLL_SEC = 90
RUNNING = {"running", "pending", "ephemeral", "active", "starting", "deployed"}
APP_ID_RE = re.compile(r"ap-[A-Za-z0-9]+")


def _run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=REPO, text=True, **kwargs)


def pull():
    LOCAL.mkdir(parents=True, exist_ok=True)
    print(f"pulling {VOLUME}:{REMOTE} -> {LOCAL}", flush=True)
    r = _run(
        ["modal", "volume", "get", "--force", VOLUME, REMOTE, str(LOCAL)],
        capture_output=True,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr or r.stdout or "volume get failed\n")
        return False
    print("pull ok", flush=True)
    return True


def _app_running(app_id: str) -> bool:
    r = _run(["modal", "app", "list", "--json"], capture_output=True, timeout=60)
    if r.returncode != 0:
        # CLI flake: assume still running so we keep polling.
        return True
    text = (r.stdout or "").strip()
    if not text:
        return False
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return app_id in (r.stdout or "") and "stop" not in (r.stdout or "").lower()
    apps = data if isinstance(data, list) else data.get("apps", data.get("items", []))
    for app in apps or []:
        if isinstance(app, dict):
            aid = str(app.get("app_id") or app.get("id") or "")
            state = str(app.get("state") or app.get("status") or "").lower()
        else:
            aid, state = str(app), ""
        if aid == app_id:
            if any(s in state for s in ("stop", "fail", "complete", "done", "timeout")):
                return False
            if not state or any(s in state for s in RUNNING):
                return True
            return False
    return False


def main():
    print("launching detached sweep...", flush=True)
    launched = _run(
        ["modal", "run", "--detach", "scripts/modal_class_struct.py", "--stage", "all"],
        capture_output=True,
    )
    combined = (launched.stdout or "") + "\n" + (launched.stderr or "")
    sys.stdout.write(launched.stdout or "")
    sys.stderr.write(launched.stderr or "")
    if launched.returncode != 0:
        print("launch failed; pulling anything already on the volume", flush=True)
        pull()
        sys.exit(launched.returncode)

    match = APP_ID_RE.search(combined)
    app_id = match.group(0) if match else None
    print(f"app_id={app_id or 'unknown'}; polling until it stops", flush=True)

    try:
        if app_id:
            # Let the app register before the first emptiness check.
            time.sleep(20)
            while _app_running(app_id):
                pull()
                time.sleep(POLL_SEC)
        else:
            print("could not parse app id; pulling once and exiting poll loop", flush=True)
    except KeyboardInterrupt:
        print("\nstopped polling (Modal sweep is still running if it was)", flush=True)
    finally:
        pull()
    print("done. files in", LOCAL, flush=True)


if __name__ == "__main__":
    main()
