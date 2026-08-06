#!/usr/bin/env python3
"""vast.ai in-container runner — the ENTRYPOINT of the vast session image (remote/Dockerfile.vast).

The rented container IS the session: this process boots sshd on internal port 22 (vast publishes it on a random external port; the cs client discovers it via the API), records state under /cs-state, idle-monitors like the other backends, and — critically — DESTROYS ITS OWN INSTANCE via the vast API on the way out. Destroy is the only thing that stops vast billing, and this process is the only component guaranteed to be around to call it; it uses the vast-injected per-instance credentials (CONTAINER_ID + CONTAINER_API_KEY).

Session parameters arrive as env vars set at instance creation (see VastBackend.submit): CS_SESSION_ID, CS_IDLE_TIMEOUT_MINUTES, CS_MAX_HOURS, CS_AUTHORIZED_KEYS_B64.
"""
import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from runner import Session, detect_cuda_version, log, monitor_loop, prepare_sshd, utcnow, wait_for_ssh_banner

PORT = 22
# After a FAILED activation, keep the instance alive this long before self-destructing: the container log (the only diagnostic that exists) is fetchable via the vast API only while the instance lives, and an agent watching the pending session gets a chance to deactivate (which salvages the log). Bounded cost: ~$0.5 at typical rates.
FAILURE_GRACE_SECONDS = 15 * 60


def self_destruct():
    import urllib.request

    iid = os.environ.get("CONTAINER_ID", "")
    key = os.environ.get("CONTAINER_API_KEY", "")
    if not iid or not key:
        log("SELF-DESTRUCT UNAVAILABLE: CONTAINER_ID/CONTAINER_API_KEY not injected — destroy the instance manually!")
        return
    for attempt in range(10):
        req = urllib.request.Request(
            f"https://console.vast.ai/api/v0/instances/{iid}/",
            data=b"{}",
            method="DELETE",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                log(f"self-destruct requested (HTTP {resp.status})")
                return
        except Exception as exc:
            log(f"self-destruct attempt {attempt + 1}/10 failed: {exc}")
            time.sleep(min(60, 2 ** attempt))
    log("SELF-DESTRUCT FAILED — the instance keeps billing until destroyed via console/API!")


class VastSession(Session):
    """Flat in-container layout — the container is the whole session, so no per-session nesting."""

    def __init__(self, session_id):
        self.base = Path("/")
        self.id = session_id
        self.dir = Path("/cs-state")
        self.config_path = self.dir / "config.json"
        self.workdir = Path("/workdir")
        self.logdir = Path("/cs-logs")
        self.home_dir = Path("/root")
        self.sshd_dir = self.dir / "sshd"
        self.helpers_dir = self.dir / "helpers"
        self.socket = self.dir / "session.sock"  # unused (direct connection, no tunnel)
        self.activity = self.dir / ".activity"


def main():
    session_id = os.environ.get("CS_SESSION_ID", "unknown")
    idle_timeout_minutes = int(os.environ.get("CS_IDLE_TIMEOUT_MINUTES") or 20)
    max_hours = float(os.environ.get("CS_MAX_HOURS") or 12)
    started = time.time()

    sess = VastSession(session_id)
    for d in (sess.dir, sess.workdir, sess.logdir):
        d.mkdir(parents=True, exist_ok=True)
    # Initial record — the client-side reconciler reads this over ssh; "active" is its readiness signal.
    sess.config_path.write_text(json.dumps({
        "session_id": session_id,
        "status": "pending",
        "idle_timeout_minutes": idle_timeout_minutes,
    }, indent=2))

    # A TERM (vast stopping the container during an API destroy) must not trigger the failure grace sleep — mark it so the finally-block distinguishes "torn down" from "broke during activation".
    signalled = {"yes": False}

    def _on_signal(signum, frame):
        signalled["yes"] = True
        sys.exit(128 + signum)

    for signo in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signo, _on_signal)

    procs = []
    activated = False
    try:
        auth_b64 = os.environ.get("CS_AUTHORIZED_KEYS_B64", "")
        if not auth_b64:
            raise SystemExit("fatal: CS_AUTHORIZED_KEYS_B64 not set — no keys to authorize")
        authorized = base64.b64decode(auth_b64).decode()

        setenv = {"CS_CUDA_VERSION": detect_cuda_version()}
        prepare_sshd(sess, PORT, "0.0.0.0", setenv,
                     authorized_keys_text=authorized, permit_root=True, conf_dir=str(sess.sshd_dir))
        Path("/run/sshd").mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(["/usr/sbin/sshd", "-f", str(sess.sshd_dir / "sshd_config"), "-D", "-e"])
        procs.append(proc)
        if not wait_for_ssh_banner(PORT, 30, procs):
            raise SystemExit("fatal: sshd did not come up on :22 within 30s")

        now = utcnow()
        sess.config_update(
            status="active",
            job_id=os.environ.get("CONTAINER_ID", ""),
            node=os.environ.get("PUBLIC_IPADDR", ""),
            sshd_port=int(os.environ.get("VAST_TCP_PORT_22") or 0) or None,
            last_activated_at=now,
            last_activity_at=now,
        )
        activated = True
        log(f"session {session_id} active (external port {os.environ.get('VAST_TCP_PORT_22', '?')})")
        reason = monitor_loop(sess, procs, PORT, idle_timeout_minutes, deadline=started + max_hours * 3600)
        log(f"shutting down: {reason}")
    except SystemExit as exc:
        if exc.code not in (0, None):
            log(str(exc.code) if isinstance(exc.code, str) else f"exiting: {exc.code}")
    finally:
        for p in procs:
            try:
                p.terminate()
            except OSError:
                pass
        if not activated and not signalled["yes"]:
            log(f"activation failed — keeping the instance for {FAILURE_GRACE_SECONDS // 60} min so the container log stays fetchable (deactivate salvages it and destroys sooner), then self-destructing")
            try:
                time.sleep(FAILURE_GRACE_SECONDS)
            except SystemExit:
                pass  # externally torn down during the grace window — proceed to the (then 404) destroy
        self_destruct()
        # If the destroy went through, vast tears this container down momentarily; nothing further to clean.


if __name__ == "__main__":
    main()
