"""compute-sessions login-node helper: all cluster-side work of one cs command, done in ONE request. The client streams this file plus a JSON request into `python3 -I` on the login node (see HostAccess.call) and reads one JSON reply after a marker line (shell-startup banners may print before it). Stdlib only, Python 3.8+: it runs with the login node's system python.

The helper owns everything that reads or writes cluster-side state: session records, the workdir manifest, command logs, and the scheduler. Record read-modify-writes take a per-session flock in the login node's local temp dir, so client writes are atomic against each other; the runner (on a compute node) writes with its own compare-and-set.
"""
import fcntl
import json
import os
import pwd
import re
import signal
import stat
import sys
import tempfile
import time
from contextlib import contextmanager

# Every request starts a fresh interpreter, so modules only some operations need (subprocess, socket, base64, shutil, concurrent.futures — the last pulls in logging) are imported where they are used.

SID_RE = re.compile(r"^[a-z0-9_]{1,64}$")
CID_RE = re.compile(r"^[A-Za-z0-9_]{1,128}$")
LIVE = ("active", "pending")
GLOB_HINT = " (wildcards are not expanded — pass the paths, or the directory that holds them)"
# zsh does not word-split an unquoted $var, so `cs logs $pair` passes "<session> <command>" as ONE argument.
SPACE_HINT = " — it is one argument containing spaces: pass the session and what follows as separate arguments (zsh does not split an unquoted $var)"
# A `.pid` heartbeat (touched every 10 s by a live command) older than this means the command died without recording an exit.
PID_STALE_SECONDS = 60
# `.exit` is written truncate-then-write (twice), and the shared FS shows compute-node writes with a delay — an empty or garbled `.exit` this young is still settling.
EXIT_SETTLE_SECONDS = 10
# squeue states after which the job holds no allocation; COMPLETING included (the runner's final write is guarded against clobbering a re-activated record).
TERMINAL_JOB_STATES = {"COMPLETED", "COMPLETING", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT"}
# Bytes of one log stream per reply (and how far back --tail looks): a bigger log arrives in several requests, so a multi-GB log never sits in the helper's memory, and a stream without line breaks (tqdm bars on stderr) cannot make --tail read it all.
READ_CAP = 8 << 20
# Bytes of a command's text returned by `cs commands`.
CMD_TEXT_CAP = 500
# A node whose GPU failed the runner's start-up check (recorded in <base>/bad-nodes.json) is left out of new activations this long — at most BAD_NODE_MAX of them (the newest), so a check that misfires everywhere cannot shut out the cluster.
BAD_NODE_SECONDS = 24 * 3600
BAD_NODE_MAX = 3
# `cs delete` moves a session dir into <base>/trash; trash entries older than this are removed by a later `cs delete`.
TRASH_KEEP_SECONDS = 7 * 24 * 3600


class Fail(Exception):
    """An error reported to the client; `kind` picks its exception type there (session → SessionError, usage → exit 2, remote → RemoteError)."""

    def __init__(self, msg, kind="session", **extra):
        super().__init__(msg)
        self.kind = kind
        self.extra = extra


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_json(path, data):
    tmp = "%s.tmp.%d.%s" % (path, os.getpid(), os.urandom(3).hex())
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def touch(path):
    with open(path, "a"):
        pass
    os.utime(path, None)


def clean_rel(rel):
    """Workdir-relative path → its components; absolute paths and `..` are refused so every path stays inside the workdir it is joined to."""
    rel = (rel or "").strip()
    if rel.startswith("/"):
        raise Fail("absolute paths not allowed: %r" % rel, kind="usage")
    parts = [p for p in rel.split("/") if p and p != "."]
    if ".." in parts:
        raise Fail("parent-directory traversal not allowed: %r" % rel, kind="usage")
    return parts


def inside(path, root):
    return path == root or path.startswith(root + "/")


def kind_of(path):
    return "dir" if os.path.isdir(path) else "file" if os.path.exists(path) else "missing"


@contextmanager
def locked(sid):
    """Per-session lock on the login node's local disk (flock on a shared FS is unreliable). The directory is private so no other user can plant the lock files."""
    d = os.path.join(tempfile.gettempdir(), "cs-locks-%d" % os.getuid())
    try:
        os.mkdir(d, 0o700)
    except FileExistsError:
        pass
    st = os.lstat(d)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise Fail("lock directory %s is not a private directory of this user" % d, kind="remote")
    path = os.path.join(d, sid + ".lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.utime(path, None)  # keeps tmp cleaners away from a lock file that is in use
        yield
    finally:
        os.close(fd)


def slurm(argv, timeout):
    """Run a scheduler command in a login shell — the scheduler tools (and the environment `sbatch --export=ALL` hands to the job) may come from the login profile. `timeout` bounds the command itself, not the profile's startup. Returns (rc, stdout, stderr); rc None = timed out."""
    import subprocess
    mark = "__CS_SLURM_%s__" % os.urandom(6).hex()
    try:
        p = subprocess.run(["bash", "-lc", 'printf "%s\\n" "$0"; exec timeout "$@"', mark, str(timeout)] + argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        p = None
    if p is None or p.returncode == 124:
        return None, "", "timed out after %ss" % timeout
    out = p.stdout.decode(errors="replace")
    i = out.find(mark + "\n")
    return p.returncode, out[i + len(mark) + 1:] if i >= 0 else out, p.stderr.decode(errors="replace")


def _start(text):
    """squeue's %S as an ISO timestamp (cluster-local time), None for N/A/unknown."""
    text = (text or "").strip()
    return text if re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$", text) else None


def job_state(job_id):
    """{state, reason, node, [time_left], [estimated_start]} with state ∈ pending/running/gone/unknown. "unknown" means the query itself failed (controller unreachable, timeout) — never to be read as gone. estimated_start is the scheduler's own forecast for a pending job (when it has one)."""
    rc, out, err = slurm(["squeue", "-h", "-j", str(job_id), "-o", "%T|%L|%R|%S"], 5)
    if rc is None:
        return {"state": "unknown", "reason": "scheduler query timed out", "node": None}
    if rc != 0:
        text = (out + err).strip()
        if "Invalid job id specified" in text:
            return {"state": "gone", "reason": None, "node": None}
        first = next((l.strip() for l in text.splitlines() if l.strip()), "squeue exit %s" % rc)
        return {"state": "unknown", "reason": "scheduler query failed: " + first, "node": None}
    text = out.strip()
    if not text:
        return {"state": "gone", "reason": None, "node": None}  # sbatch returns only once the job is queued, so an empty result means finished
    state, time_left, col, start = (text.splitlines()[-1].split("|", 3) + ["", "", ""])[:4]
    if state == "RUNNING":
        return {"state": "running", "reason": None, "node": col, "time_left": time_left or None}
    if state in ("PENDING", "CONFIGURING"):
        reason = col.strip("()")  # squeue prints a pending reason in parentheses, "(None)" when there is none
        return {"state": "pending", "reason": reason if reason not in ("", "None") else None, "node": None, "time_left": time_left or None, "estimated_start": _start(start)}
    if state in TERMINAL_JOB_STATES:
        return {"state": "gone", "reason": "%s: %s" % (state, col) if col else state, "node": None}
    # SUSPENDED, REQUEUED, … still hold or will hold an allocation: pending, so a submit refuses to duplicate it.
    return {"state": "pending", "reason": "%s: %s" % (state, col), "node": None}


def session_jobs(sid):
    """Job ids the scheduler still holds for this session (its jobs are named cs-<sid>), or None when the query failed. Finds jobs the record does not know about too — e.g. a submit whose answer was lost."""
    rc, out, err = slurm(["squeue", "-h", "-u", pwd.getpwuid(os.getuid()).pw_name, "-n", "cs-" + sid, "-o", "%i|%T"], 10)
    if rc != 0:
        return None
    jobs = []
    for line in out.splitlines():
        jid, _, state = line.strip().partition("|")
        if jid.isdigit() and state not in TERMINAL_JOB_STATES:
            jobs.append(jid)
    return jobs


def cancel(sid):
    """Cancel the session's jobs by name, so a job the record lost track of goes too. No job id on the command line: scancel would then cancel only the listed ids that also match the name."""
    slurm(["scancel", "-u", pwd.getpwuid(os.getuid()).pw_name, "-n", "cs-" + sid], 30)


class Ctx:
    def __init__(self, req):
        self.req = req
        self.base = os.path.expanduser(req["base"]).rstrip("/") or "/"
        self.root = self.base + "/sessions"

    def sdir(self, sid):
        if not SID_RE.match(sid):
            raise Fail("invalid session id: %r" % sid, kind="usage")
        return "%s/%s" % (self.root, sid)

    def record_path(self, sid):
        return self.sdir(sid) + "/config.json"

    def info(self, sid):
        d = read_json(self.record_path(sid))
        if d is None:
            raise Fail("session %r not found — `cs list` shows known sessions" % sid)
        return d

    def write_info(self, sid, info):
        write_json(self.record_path(sid), info)

    def ids(self):
        try:
            return [n for n in os.listdir(self.root) if SID_RE.match(n)]
        except FileNotFoundError:
            return []

    def records(self, ids=None):
        """Records read in parallel: per-file latency dominates on a shared FS."""
        ids = self.ids() if ids is None else ids
        if not ids:
            return []
        if len(ids) == 1:
            recs = [read_json(self.record_path(ids[0]))]
            return [r for r in recs if isinstance(r, dict) and r.get("session_id")]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(16, len(ids))) as ex:
            recs = list(ex.map(lambda s: read_json(self.record_path(s)), ids))
        return [r for r in recs if isinstance(r, dict) and r.get("session_id")]

    def touch_activity(self, sid):
        """The idle monitor counts a fresh `.activity` mtime as activity — for workdir file ops and container commands. Reading logs/records never touches it: that must not hold hardware."""
        try:
            touch(self.sdir(sid) + "/.activity")
        except OSError:
            pass

    def resolve(self):
        """The session named by the request's first arg — always its full id: no prefixes, no inference from the cwd. Returns (sid, record, rest args, note) — note is set when the session belongs to another project (other than one in a parent directory of the cwd)."""
        args = list(self.req.get("args") or [])
        token = args[0].strip() if args else ""
        if not token:
            raise Fail("no session given — pass its full id first (`cs list` shows them)", kind="usage")
        rec = read_json(self.record_path(token)) if SID_RE.match(token) else None
        if rec is None:
            raise Fail("no session %r" % token + self.miss_hint(token), kind="usage")
        return token, rec, args[1:], _note(rec, self.req.get("project_id"), self.req.get("cwd") or "")

    def miss_hint(self, token):
        """Why a session token named nothing: one argument with spaces, a shortened id (ids are needed in full), or not a session id at all."""
        if " " in token:
            return SPACE_HINT
        near = sorted(i for i in self.ids() if _starts(i, token, "_"))
        if near:
            return " — session ids are needed in full: did you mean %s?" % " or ".join(near[:3])
        if not token.startswith((self.req.get("prefix") or "") + "_"):
            return " — the first argument is the session's full id (`cs list` shows them)"
        return " — `cs list` shows known sessions"

    def exact(self):
        sid = self.req["sid"]
        return sid, self.info(sid)


def _starts(full, token, sep):
    """Whether `token` is a shortened form of the id `full`: a prefix of it, or of its part after `sep`."""
    return full != token and (full.startswith(token) or full.partition(sep)[2].startswith(token))


def _note(rec, pid, cwd):
    path = rec.get("project_path") or ""
    if rec.get("project_id") == pid or (path and path != cwd and inside(cwd, path)):
        return None
    return path or rec.get("project_id")


def _reachable(info):
    return info.get("status") == "active" and bool(info.get("sshd_port"))


def _container_target(ctx, sid, info, out):
    if _reachable(info):
        out["socket"] = ctx.sdir(sid) + "/session.sock"


# ---------- command logs ----------

def open_log(path):
    """(binary file, stat) of a file in a session's logs dir, opened without following symlinks: the dir is writable from inside the container, and a planted `cmd_x.out -> ~/.ssh/id_rsa` must not make the login node read the target. Anything but a regular file (a FIFO would block the read) raises OSError."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise OSError("not a regular file: %s" % path)
    return os.fdopen(fd, "rb"), st


def read_exit(path):
    """(exit code or None, mtime) of an `.exit` file; FileNotFoundError when there is none. Its contents are not trusted."""
    try:
        fh, st = open_log(path)
    except FileNotFoundError:
        raise
    except OSError:
        return None, 0.0
    with fh:
        text = fh.read(64).decode(errors="replace").strip()
    return (int(text) if text.lstrip("-").isdigit() else None), st.st_mtime


def activation_mark(ctx, sid, info):
    """When the session's current container started, as the mtime of its sshd pidfile — written into the session dir from the same node as the commands' `.start` files, so the two compare on one clock. A command that started earlier belongs to an earlier activation and cannot be running. None: the session is not active, so no command can be running; 0: unknown."""
    if info.get("status") != "active":
        return None
    try:
        return os.lstat(ctx.sdir(sid) + "/.sshd/sshd.pid").st_mtime
    except OSError:
        return 0


def _dead_with_session(base, mark):
    """Whether a command without an exit died with an earlier activation (or with the session) — its heartbeat may still look fresh for up to a minute."""
    if mark is None:
        return True
    try:
        return bool(mark) and os.lstat(base + ".start").st_mtime < mark
    except OSError:
        return False


def cmd_status(base, mark=0):
    """(status, exit_code) of one command from its log files: exited (`.exit` recorded), running (fresh `.pid` heartbeat in the current activation), killed (heartbeat went stale without an exit: SIGKILL/OOM — or the activation it ran in ended, see activation_mark), missing."""
    now = time.time()
    try:
        code, mtime = read_exit(base + ".exit")
    except FileNotFoundError:
        pass
    else:
        if code is None and now - mtime < EXIT_SETTLE_SECONDS:
            return "running", None
        return "exited", code
    try:
        age = now - os.lstat(base + ".pid").st_mtime
    except OSError:
        return ("queued" if os.path.lexists(base + ".queued") else "missing"), None
    return ("running" if age <= PID_STALE_SECONDS and not _dead_with_session(base, mark) else "killed"), None


def command_rows(logdir, mark=0):
    """All recorded commands, newest first: {command_id, status, [exit_code, started_at, exited_at]}. started_at comes from `.start` (written once at launch) — the `.pid` mtime is the heartbeat. Internal `cmd_venvsnap_*` housekeeping is hidden. `mark`: see activation_mark."""
    now = time.time()
    rows = {}
    try:
        names = os.listdir(logdir)
    except FileNotFoundError:
        names = []
    for name in names:
        if not name.startswith("cmd_") or name.startswith("cmd_venvsnap_"):
            continue
        base, _, ext = name.rpartition(".")
        if ext not in ("start", "pid", "exit", "queued", "wait"):
            continue
        path = os.path.join(logdir, name)
        try:
            mtime = os.lstat(path).st_mtime
        except OSError:
            continue
        row = rows.setdefault(base, {"command_id": base, "status": "running", "exit_code": None, "started_at": None, "exited_at": None, "_pid_age": None, "_exit_age": None, "_start": None, "_queued": None, "_wait": False})
        if ext == "wait":
            row["_wait"] = True
            continue
        if ext == "queued":
            row["_queued"] = mtime
        elif ext == "start":
            row["started_at"] = row["_start"] = mtime
        elif ext == "pid":
            row["_pid_age"] = now - mtime
        else:
            try:
                row["exit_code"] = read_exit(path)[0]
            except OSError:
                continue
            row["exited_at"] = mtime
            row["_exit_age"] = now - mtime
    for row in rows.values():
        pid_age, exit_age, start, queued, wait = row.pop("_pid_age"), row.pop("_exit_age"), row.pop("_start"), row.pop("_queued"), row.pop("_wait")
        if pid_age is None and exit_age is None and queued is not None:
            row["status"], row["queued_at"] = "queued", queued
        elif exit_age is not None and (row["exit_code"] is not None or exit_age >= EXIT_SETTLE_SECONDS):
            row["status"] = "exited"
        elif exit_age is None and (pid_age is None or pid_age > PID_STALE_SECONDS or mark is None or (mark and start is not None and start < mark)):
            row["status"] = "killed"
        else:
            row["exited_at"] = None
            if wait:
                row["status"] = "waiting"  # cs run --after: started, but holding the command until the ones it waits for succeed
    ordered = sorted(rows.values(), key=lambda r: r["started_at"] or r["exited_at"] or r.get("queued_at") or 0, reverse=True)
    for r in ordered:
        for k in ("started_at", "exited_at", "queued_at"):
            if r.get(k) is not None:
                r[k] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r[k]))
    return [{k: v for k, v in r.items() if v is not None} for r in ordered]


def read_slice(path, off, max_lines):
    """(bytes from `off`, new offset, whether the file has more): at most READ_CAP bytes; with max_lines only the last N lines of the slice, looking back at most READ_CAP bytes. The offset never moves back: the shared FS can show less than a client already received from the container."""
    try:
        fh, st = open_log(path)
    except OSError:
        return b"", off, False
    size = st.st_size
    with fh:
        if size <= off:
            return b"", off, False
        if not max_lines:
            fh.seek(off)
            n = min(size - off, READ_CAP)
            return fh.read(n), off + n, off + n < size
        start = max(off, size - READ_CAP)
        fh.seek(start)
        buf = fh.read(size - start)
    # The last N line breaks, plus the final one when the slice ends with one.
    idx = len(buf)
    for _ in range(max_lines + buf.endswith(b"\n")):
        idx = buf.rfind(b"\n", 0, idx)
        if idx < 0:
            break
    return buf[idx + 1:], size, False


# ---------- ops ----------

def op_list(ctx):
    """Session records; a pending one gets the scheduler's start forecast (one scheduler query, only when something is pending)."""
    req = ctx.req
    recs = ctx.records()
    if not req.get("all"):
        # Same-directory sessions under another project id are kept too: the client flags them (the id derives from the git remote and changes when one is added).
        recs = [r for r in recs if r.get("project_id") == req.get("project_id") or r.get("project_path") == req.get("cwd")]
    if any(r.get("status") == "pending" and r.get("job_id") for r in recs):
        rc, out, _ = slurm(["squeue", "-h", "-u", pwd.getpwuid(os.getuid()).pw_name, "-t", "PENDING", "-o", "%i|%S"], 10)
        starts = dict(l.strip().split("|", 1) for l in out.splitlines() if "|" in l) if rc == 0 else {}
        for r in recs:
            if r.get("status") == "pending" and _start(starts.get(str(r.get("job_id")))):
                r["estimated_start"] = _start(starts[str(r["job_id"])])
    return {"sessions": recs}


def reconcile(ctx, sid, info):
    """(record, job state). A job that is gone while the record still says pending/active (the runner died without cleanup) is written back as failed/inactive — compare-and-set: a concurrent activate/deactivate owns a record that has moved on."""
    job = job_state(info["job_id"])
    target = {"pending": "failed", "active": "inactive"}.get(info.get("status"))
    if job["state"] == "gone" and target:
        with locked(sid):
            cur = ctx.info(sid)
            if cur.get("job_id") == info["job_id"] and cur.get("status") == info["status"]:
                cur.update(status=target, last_deactivated_at=now_iso(), stop_reason=cur.get("stop_reason") or "the job ended without the runner's cleanup (node failure or hard kill)")
                ctx.write_info(sid, cur)
            info = cur
    return info, job


def op_show(ctx):
    """Record, optionally the job state (a gone job is reconciled), the failure log tail, and the sshd banner check."""
    req = ctx.req
    sid, info, rest, note = ctx.resolve()
    out = {"sid": sid, "note": note}
    if req.get("squeue") and info.get("job_id"):
        info, out["job"] = reconcile(ctx, sid, info)
    out["info"] = info
    if info.get("status") == "failed":
        logdir = ctx.sdir(sid) + "/logs"
        try:
            runner_logs = sorted((n for n in os.listdir(logdir) if n.startswith("runner")), key=lambda n: os.lstat(os.path.join(logdir, n)).st_mtime)
        except OSError:
            runner_logs = []
        if runner_logs:
            out["failure_log_tail"] = read_slice(os.path.join(logdir, runner_logs[-1]), 0, 30)[0].decode(errors="replace")
    if req.get("probe") and info.get("status") == "active":
        out["health"] = probe(ctx.sdir(sid) + "/session.sock")
    return out


def probe(sock_path):
    """Read the SSH banner through the session's reverse-tunnel socket: proves both the tunnel and the container sshd are alive."""
    import socket
    if not os.path.exists(sock_path):
        return {"reachable": False, "detail": "endpoint not created yet (session may still be starting)"}
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(3)
    try:
        s.connect(sock_path)
        banner = s.recv(256)
    except OSError as exc:
        return {"reachable": False, "detail": str(exc)}
    finally:
        s.close()
    if banner.startswith(b"SSH-"):
        return {"reachable": True}
    return {"reachable": False, "detail": banner.decode(errors="replace").strip() or "no response"}


def resolve_cid(ctx, sid, token):
    """A command id named by the user — its full id."""
    if not CID_RE.match(token):
        raise Fail("invalid command id: %r" % token + (SPACE_HINT if " " in token.strip() else ""), kind="usage")
    logdir = ctx.sdir(sid) + "/logs"
    if os.path.lexists("%s/%s.pid" % (logdir, token)) or os.path.lexists("%s/%s.queued" % (logdir, token)):
        return token
    try:
        names = os.listdir(logdir)
    except FileNotFoundError:
        names = []
    near = sorted({n.rpartition(".")[0] for n in names if n.endswith((".pid", ".queued")) and not n.startswith("cmd_venvsnap_") and _starts(n.rpartition(".")[0], token, "cmd_")})
    hint = "command ids are needed in full: did you mean %s?" % " or ".join(near[:3]) if near else "`cs commands %s` lists them (the session goes first: cs <cmd> SESSION COMMAND_ID)" % sid
    raise Fail("no command %r in session %s — %s" % (token, sid, hint), kind="usage")


def op_session(ctx):
    """Resolve, optionally mark activity, and return the record plus — when the container is reachable — the socket to it. With `cid`, a first remaining arg is a command id and must exist; so must `after`'s (`cs run --after`)."""
    sid, info, rest, note = ctx.resolve()
    if ctx.req.get("cid") and rest:
        resolve_cid(ctx, sid, rest[0])
    after = [resolve_cid(ctx, sid, t) for t in ctx.req.get("after") or []]
    if after:
        mark = activation_mark(ctx, sid, info)
        for cid in after:
            status, code = cmd_status("%s/logs/%s" % (ctx.sdir(sid), cid), mark)
            if status == "killed" or (status == "exited" and code != 0):
                raise Fail("%s already %s — a command run --after it would never start" % (cid, "exited %s" % code if code is not None else "ended without an exit code"), kind="usage")
    if ctx.req.get("touch"):
        ctx.touch_activity(sid)
    out = {"sid": sid, "note": note, "rest": rest, "info": info, "after": after}
    _container_target(ctx, sid, info, out)
    return out


def op_logs(ctx):
    """A command's state and — unless `status_only` — its output from `since` (READ_CAP per stream; `more` asks for another request). With `squeue`, a session whose job is gone is reconciled first, so a stream to a dead session stops retrying. With `resolve` (a user-given id), an unknown id is an error; internal follow-up reads get status "missing" instead."""
    import base64
    req = ctx.req
    sid, info, rest, note = ctx.resolve()
    if len(rest) != 1:
        raise Fail("usage: cs logs SESSION COMMAND_ID" + ("" if rest else " — `cs commands %s` lists the command ids" % sid), kind="usage")
    if req.get("squeue") and info.get("job_id"):
        info = reconcile(ctx, sid, info)[0]
    logdir = ctx.sdir(sid) + "/logs"
    out = {"sid": sid, "note": note, "session_status": info.get("status")}
    mark = activation_mark(ctx, sid, info)
    cid = rest[0]
    if req.get("resolve"):
        resolve_cid(ctx, sid, cid)
    elif not CID_RE.match(cid):
        raise Fail("invalid command id: %r" % cid, kind="usage")
    base = "%s/%s" % (logdir, cid)
    status, code = cmd_status(base, mark)
    out.update(cid=cid, status=status)
    if status == "killed" and mark is not None and _dead_with_session(base, mark):
        out["earlier_activation"] = True
    if code is not None:
        out["exit_code"] = code
    _container_target(ctx, sid, info, out)
    if req.get("status_only"):
        return out
    oo, eo = req.get("since") or [0, 0]
    o, oo, more_o = read_slice(base + ".out", oo, req.get("max_lines"))
    e, eo, more_e = read_slice(base + ".err", eo, req.get("max_lines"))
    out.update(out=base64.b64encode(o).decode(), err=base64.b64encode(e).decode(), offsets=[oo, eo])
    if more_o or more_e:
        out["more"] = True
    return out


def read_cmd(path):
    """The start of a command's text (`.cmd`, written at launch; absent for commands launched by older clients)."""
    try:
        fh, _ = open_log(path)
    except OSError:
        return None
    with fh:
        return fh.read(CMD_TEXT_CAP).decode(errors="replace")


def op_commands(ctx):
    req = ctx.req
    sid, info, rest, note = ctx.resolve()
    logdir = ctx.sdir(sid) + "/logs"
    rows = command_rows(logdir, activation_mark(ctx, sid, info))
    limit, offset = max(0, int(req.get("limit") or 0)), max(0, int(req.get("offset") or 0))
    page = rows[offset:offset + limit] if limit else rows[offset:]
    for r in page:
        text = read_cmd("%s/%s.cmd" % (logdir, r["command_id"]))
        if text is not None:
            r["command"] = text
    return {"sid": sid, "note": note, "commands": page, "total": len(rows)}


def resolve_dest(src_base, dest, src_is_dir, src_trailing_slash, dest_trailing_slash, dest_is_dir):
    """Where a transfer lands (upload, download, copy), by cp/rsync conventions: a dest ending in `/` or an existing directory receives the source INSIDE it (dest/<basename>); otherwise the dest names the result. Directory contents merge directly into dest only when the SOURCE ends with `/` (rsync's src/ idiom; also `.`/empty), or when dest is an existing dir of the source's own name (a repeated `upload data data` stays idempotent instead of nesting data/data). Uploads and copies create missing destination directories (the remote side is cheap to inspect with `cs ls`); downloads need the local ones to exist — local paths resolve against the project, not the cwd, and a silently created tree would hide a wrong base."""
    if src_is_dir and src_trailing_slash:
        return dest  # explicit contents-merge
    if dest_trailing_slash or (dest_is_dir and not (src_is_dir and src_base == dest.rstrip("/").rsplit("/", 1)[-1])):
        return "%s/%s" % (dest.rstrip("/"), src_base) if dest.rstrip("/") else src_base
    return dest


def _workdir_target(ctx, sid, rel):
    """(clean rel, real path) of a workdir path; refuses a path whose real location is outside the workdir — `[ -d ]`, `ls` and rsync follow a symlinked directory component, so a link planted from inside the container (`workdir/evil -> ~/.ssh`) would otherwise turn file ops into arbitrary reads/writes of the cluster user's files."""
    parts = clean_rel(rel)
    wd = os.path.realpath(ctx.sdir(sid) + "/workdir")
    real = os.path.realpath(os.path.join(wd, *parts))
    if not inside(real, wd):
        raise Fail("remote path %r resolves outside the session workdir (symlink) — refusing to operate" % ("/".join(parts) or "."), kind="usage")
    return "/".join(parts), real, wd


def _drop_queued(ctx, sid, cid=None):
    """Take queued commands (one, or all — deactivate drops them all) out of the session's queue; returns their ids. The rename is the claim: the runner launching the same entry renames it first or not at all."""
    qdir = ctx.sdir(sid) + "/queue"
    try:
        names = sorted(os.listdir(qdir))
    except FileNotFoundError:
        return []
    dropped = []
    for name in names:
        if not CID_RE.match(name) or (cid and name != cid):
            continue
        try:
            os.rename("%s/%s" % (qdir, name), "%s/.dropped.%s" % (qdir, name))
        except FileNotFoundError:
            continue
        os.unlink("%s/.dropped.%s" % (qdir, name))
        for ext in (".queued", ".cmd"):
            try:
                os.unlink("%s/logs/%s%s" % (ctx.sdir(sid), name, ext))
            except FileNotFoundError:
                pass
        dropped.append(name)
    return dropped


def op_queue(ctx):
    """Queue a command on a pending session: the runner launches it (the client-built launch script) once the session is active. Refused with `queued: False` when the session is no longer pending — the client then launches it itself."""
    import base64
    req = ctx.req
    sid, info = ctx.exact()
    cid = req["cid"]
    if not CID_RE.match(cid):
        raise Fail("invalid command id: %r" % cid, kind="usage")
    with locked(sid):
        info = ctx.info(sid)
        if info.get("status") != "pending":
            return {"sid": sid, "queued": False, "info": info}
        qdir, logdir = ctx.sdir(sid) + "/queue", ctx.sdir(sid) + "/logs"
        os.makedirs(qdir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        with open("%s/%s.cmd" % (logdir, cid), "w") as fh:
            fh.write(req["command"])
        touch("%s/%s.queued" % (logdir, cid))
        with open("%s/.%s.tmp" % (qdir, cid), "wb") as fh:
            fh.write(base64.b64decode(req["script"]))
        os.replace("%s/.%s.tmp" % (qdir, cid), "%s/%s" % (qdir, cid))
    return {"sid": sid, "queued": True, "info": info}


def op_dequeue(ctx):
    """`cs kill` of a queued command."""
    sid, info, rest, note = ctx.resolve()
    if len(rest) != 1:
        raise Fail("usage: cs kill SESSION COMMAND_ID", kind="usage")
    with locked(sid):
        return {"sid": sid, "note": note, "info": info, "dropped": _drop_queued(ctx, sid, rest[0])}


def op_ls(ctx):
    sid, info, rest, note = ctx.resolve()
    if len(rest) > 1:
        raise Fail("too many arguments — usage: cs ls SESSION [PATH]", kind="usage")
    ctx.touch_activity(sid)
    rel, real, _ = _workdir_target(ctx, sid, rest[0] if rest else "")
    if not os.path.lexists(real):
        raise Fail("path not found in session %s workdir: %r%s" % (sid, rel or ".", GLOB_HINT if any(c in rel for c in "*?[") else ""), kind="usage")
    # A file is listed by its name, never the cluster-side absolute path.
    if os.path.isdir(real) and not os.path.islink(real):
        try:
            names = sorted(os.listdir(real))
        except OSError as exc:
            raise Fail("cannot list %r: %s" % (rel or ".", exc.strerror))
        lines = [_ls_line(os.path.join(real, n), n) for n in names] or ["(empty)"]
    else:
        lines = [_ls_line(real, os.path.basename(real))]
    return {"sid": sid, "note": note, "listing": "".join(l + "\n" for l in lines)}


def _ls_line(path, name):
    """One listing line: size (human, files only), modification time (login node's local time), name — directories end in '/', links show their target."""
    try:
        st = os.lstat(path)
    except OSError:
        return "%8s  %16s  %s" % ("?", "?", name)
    size, shown = "-", name
    if stat.S_ISDIR(st.st_mode):
        shown += "/"
    elif stat.S_ISLNK(st.st_mode):
        try:
            shown += " -> " + os.readlink(path)
        except OSError:
            pass
    else:
        n, unit = float(st.st_size), "B"
        for unit in ("B", "K", "M", "G", "T"):
            if n < 1024 or unit == "T":
                break
            n /= 1024
        size = "%d%s" % (n, unit) if unit == "B" else "%.1f%s" % (n, unit)
    return "%8s  %s  %s" % (size, time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)), shown)


def op_path_check(ctx):
    """For upload/download: kind of a workdir path, the kind of `<path>/<child>` (where a place-inside upload lands), and the real path to transfer to/from — all symlink-escape checked. With `make_dirs` (uploads), a missing destination's directories are created first: the path itself when it ends in '/', else its parent."""
    sid, _ = ctx.exact()
    path = ctx.req.get("path") or ""
    rel, real, wd = _workdir_target(ctx, sid, path)
    if ctx.req.get("make_dirs") and rel and not os.path.lexists(real):
        os.makedirs(real if path.rstrip().endswith("/") else os.path.dirname(real), exist_ok=True)
    child = ctx.req.get("child")
    child_kind = None
    if child:
        creal = os.path.realpath(os.path.join(real, child))
        if not inside(creal, wd):
            raise Fail("remote destination %s/%s resolves outside the session workdir (symlink) — refusing to operate" % (rel or ".", child), kind="usage")
        child_kind = kind_of(creal)
    return {"rel": rel, "real": real, "workdir": wd, "kind": kind_of(real), "child_kind": child_kind}


def _resolve_token(ctx, token):
    """(sid, record, note) of a session named by its id — for commands that name two sessions."""
    saved = ctx.req.get("args")
    ctx.req["args"] = [token]
    try:
        sid, rec, _, note = ctx.resolve()
    finally:
        ctx.req["args"] = saved
    return sid, rec, note


def op_copy(ctx):
    """Copy workdir paths of one session into another session's workdir (or elsewhere in the same one), on the cluster: cp conventions as for upload (resolve_dest), missing destination directories created, both sides symlink-escape checked. Symlinks are copied as links."""
    import subprocess
    req = ctx.req
    src, _, note_s = _resolve_token(ctx, req["src"])
    dst, _, note_d = _resolve_token(ctx, req["dst"])
    ctx.touch_activity(src)
    ctx.touch_activity(dst)
    copied = []
    for path in req["paths"]:
        srel, sreal, _ = _workdir_target(ctx, src, path)
        if not os.path.lexists(sreal):
            raise Fail("path not found in session %s workdir: %r%s" % (src, srel or ".", GLOB_HINT if any(c in srel for c in "*?[") else ""), kind="usage")
        src_is_dir = os.path.isdir(sreal)
        name = srel.rsplit("/", 1)[-1] if srel else "workdir"
        dest_in = (req.get("dest") or "").strip()
        if not dest_in or dest_in == ".":
            dest_in = name if srel else "./"
        drel, dreal, _ = _workdir_target(ctx, dst, dest_in)
        if drel and not os.path.lexists(dreal):
            os.makedirs(dreal if dest_in.endswith("/") else os.path.dirname(dreal), exist_ok=True)
        dest = resolve_dest(name, drel, src_is_dir, path.rstrip().endswith("/") or not srel, dest_in.endswith("/"), kind_of(dreal) == "dir")
        trel, treal, _ = _workdir_target(ctx, dst, dest)
        if src == dst and src_is_dir and inside(treal, sreal):
            raise Fail("cannot copy %r into itself (%r)" % (srel or ".", trel or "."), kind="usage")
        # rsync writes each file to a temp name and renames it into place: a symlink planted in the destination (from inside its container) is replaced, never written through — `cp` would follow it out of the workdir.
        if src_is_dir:
            if os.path.isfile(treal):
                raise Fail("destination %r exists as a file but the source is a directory — pick another path" % trel, kind="usage")
            os.makedirs(treal, exist_ok=True)
            argv = ["rsync", "-a", sreal + "/", treal + "/"]
        else:
            if os.path.isdir(treal):
                raise Fail("destination %r exists as a directory but the source is a file — pick another path" % trel, kind="usage")
            argv = ["rsync", "-a", sreal, treal]
        p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise Fail("copy of %r failed: %s" % (srel or ".", p.stderr.decode(errors="replace").strip()), kind="remote")
        copied.append([srel or ".", (trel or ".") + ("/" if src_is_dir else "")])
    return {"sid": dst, "src": src, "note": note_s or note_d, "copied": copied}


def op_delete(ctx):
    """Move stopped sessions (workdir, logs, record) into <base>/trash; trash entries older than TRASH_KEEP_SECONDS are removed in the background. Only exact ids: deletion is never inferred."""
    import subprocess
    trash = ctx.base + "/trash"
    deleted = []
    for sid in ctx.req["sids"]:
        with locked(sid):
            info = ctx.info(sid)
            if info.get("status") in LIVE:
                raise Fail("session %s is %s — `cs deactivate %s` first" % (sid, info["status"], sid), kind="usage")
            jobs = session_jobs(sid)
            if jobs is None or jobs:
                raise Fail("session %s %s — retry in a moment" % (sid, "still has a job" if jobs else "cannot be checked (the scheduler query failed)"))
            os.makedirs(trash, exist_ok=True)
            os.rename(ctx.sdir(sid), "%s/%s.%d" % (trash, sid, int(time.time())))
        deleted.append(sid)
    try:
        names = os.listdir(trash)
    except FileNotFoundError:
        names = []
    old = ["%s/%s" % (trash, n) for n in names if re.match(r"^[a-z0-9_]+\.\d+$", n) and time.time() - int(n.rpartition(".")[2]) > TRASH_KEEP_SECONDS]
    if old:
        # Removing TBs from a shared FS takes long; the request does not wait for it.
        subprocess.Popen(["rm", "-rf", "--"] + old, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"deleted": deleted, "trash": trash, "keep_days": TRASH_KEEP_SECONDS // 86400}


def _check_cap(ctx, project_id, cap):
    if cap and cap > 0:
        live = [r for r in ctx.records() if r.get("project_id") == project_id and r.get("status") in LIVE]
        if len(live) >= cap:
            raise Fail("repo %r already has %d active/pending session(s); limit is %d — deactivate one of yours (parallel agents may own others: %s) or raise max_sessions_per_repo in the config" % (project_id, len(live), cap, ", ".join(r["session_id"] for r in live[:4])), kind="usage")


def _new_session(ctx, record):
    """Called with the new session's lock held."""
    sid = record["session_id"]
    os.makedirs(ctx.root, exist_ok=True)
    try:
        os.mkdir(ctx.sdir(sid))
    except FileExistsError:
        raise Fail("session %s was aborted before it was created" % sid)
    for d in ("workdir", "logs"):
        os.mkdir(ctx.sdir(sid) + "/" + d)
    ctx.write_info(sid, record)
    return sid


def op_create(ctx):
    record = ctx.req["record"]
    _check_cap(ctx, record["project_id"], ctx.req.get("cap"))
    with locked(record["session_id"]):
        sid = _new_session(ctx, record)
    return {"sid": sid, "workdir": ctx.sdir(sid) + "/workdir"}


def op_clone(ctx):
    """New session for the source's project, seeded with a server-side copy of its workdir and sync manifest (logs are the source's history and stay behind)."""
    src, src_info, rest, note = ctx.resolve()
    # The source's resources and idle timeout carry over: a clone without resource flags reuses them.
    record = dict(ctx.req["record"], project_id=src_info["project_id"], project_path=src_info.get("project_path", ""), resources=src_info.get("resources") or {}, idle_timeout_minutes=src_info.get("idle_timeout_minutes") or 0)
    _check_cap(ctx, record["project_id"], ctx.req.get("cap"))
    sid = record["session_id"]
    import shutil
    import subprocess
    # Held across the copy, so an abort of an interrupted clone waits for it instead of racing it.
    with locked(sid):
        _new_session(ctx, record)
        p = subprocess.run(["cp", "-a", ctx.sdir(src) + "/workdir/.", ctx.sdir(sid) + "/workdir/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            ctx.write_info(sid, dict(record, status="failed", last_deactivated_at=now_iso()))
            raise Fail("could not copy the workdir of %s: %s" % (src, p.stderr.decode(errors="replace").strip()), kind="remote")
        try:
            shutil.copyfile(ctx.sdir(src) + "/.sync-manifest", ctx.sdir(sid) + "/.sync-manifest")
        except FileNotFoundError:
            pass
    return {"sid": sid, "note": note}


def op_submit(ctx):
    """Start an activation: refuse while the session still has a job (looked up by its name, so one the record lost track of counts too), write the pending record, sbatch, stamp the job id. Without `resources` (no resource flag given) the previous activation's resources are reused; without an idle timeout, the previous one (else 20 min). The lock is held throughout, so an abort/deactivate arriving mid-submit (Ctrl-C during create: this helper keeps running after the client is gone) waits and then sees — and cancels — the right job."""
    req = ctx.req
    sid, info, rest, note = ctx.resolve()
    with locked(sid):
        info = ctx.info(sid)
        jobs = session_jobs(sid)
        if jobs is None:
            raise Fail("cannot tell whether session %s still has a job (the scheduler query failed) — retry in a moment, or `cs deactivate` first" % sid)
        if jobs:
            raise Fail("session %s is already active (job_id=%s); deactivate it first, then activate to restart with new resources." % (sid, jobs[0]), kind="usage")
        r = req.get("resources") or info.get("resources")
        if not r:
            raise Fail("session %s has no previous resources to reuse — pass --partition (and --gpus/--mem as needed)" % sid, kind="usage")
        if req.get("manifest") is not None:
            write_lines(ctx.sdir(sid) + "/.sync-manifest", req["manifest"])
        prev = info.get("status"), info.get("job_id")
        # The previous activation's runtime state must not show through while pending — nor its job id: show() would find that job gone and fail the fresh session.
        info.update(resources=r, idle_timeout_minutes=req.get("idle_timeout_minutes") or info.get("idle_timeout_minutes") or 20, image=req["image"], status="pending", job_id=None, node=None, sshd_port=None, gpu=None, last_activity_at=None, job_started_at=None, stop_reason=None, busy=None, submitted_at=now_iso())
        ctx.write_info(sid, info)  # before sbatch: the runner reads the record as its first act
        logs = ctx.sdir(sid) + "/logs"
        os.makedirs(logs, exist_ok=True)  # the scheduler opens --output before the runner runs; a record an abort wrote has no logs dir
        flags = ["--export=ALL", "--parsable", "--job-name=cs-" + sid, "--partition=" + r["partition"]]
        exclude = list(req.get("exclude_nodes") or [])
        if r.get("gpus"):
            exclude += [n for n in bad_nodes(ctx) if n not in exclude]
        if exclude:
            flags.append("--exclude=" + ",".join(exclude))
        if r.get("gpus"):
            flags.append("--gpus-per-node=%d" % r["gpus"])
        if r.get("gpu_type"):
            flags.append("--constraint=" + r["gpu_type"])
        if r.get("cpus"):
            flags.append("--cpus-per-task=%d" % r["cpus"])
        flags += ["--mem=%dG" % r["mem"], "--output=%s/runner-%%j.out" % logs]
        rc, out, err = slurm(["sbatch"] + flags + [ctx.base + "/runner.py", "--base", ctx.base, "--session-id", sid, "--login", req["login"]], 120)
        lines = out.strip().splitlines()
        job_id = lines[-1].split(";")[0].strip() if rc == 0 and lines else ""
        if rc is None:
            # No answer: the job may have been queued all the same.
            found = session_jobs(sid)
            if found is None:
                raise Fail("the scheduler did not answer the submission, and whether it went through is unknown — `cs show %s` will tell; `cs deactivate %s` cancels it either way" % (sid, sid), kind="remote")
            job_id = found[0] if found else ""
        cur = ctx.info(sid)
        if not job_id.isdigit():
            # Put the record back: a pending record without a job is a state nothing reconciles.
            cur.update(status=prev[0], job_id=prev[1])
            ctx.write_info(sid, cur)
            raise Fail("the scheduler refused the session: %s" % ((err or out).strip() or "exit %s" % rc), kind="remote")
        # Only the job id, and only while pending: a fast allocation can have the runner flip the record to active (writing the id itself) already.
        if cur.get("status") == "pending" and not cur.get("job_id"):
            cur["job_id"] = job_id
            ctx.write_info(sid, cur)
    return {"sid": sid, "note": note, "info": cur}


def bad_nodes(ctx):
    """Nodes whose GPU failed a runner's start-up check within BAD_NODE_SECONDS (the runner records them)."""
    d = read_json(ctx.base + "/bad-nodes.json") or {}
    now = time.time()
    recent = [(float(e.get("at") or 0), n) for n, e in d.items() if isinstance(e, dict) and now - float(e.get("at") or 0) < BAD_NODE_SECONDS and re.match(r"^[A-Za-z0-9._-]+$", n)]
    return sorted(n for _, n in sorted(recent, reverse=True)[:BAD_NODE_MAX])


def op_abort(ctx):
    """Mark an interrupted create/clone failed (cancelling its job if it got one), so it never lingers as a jobless pending record. When the create/clone request has not written the record yet (it waits for this lock, its client already gone), a failed record takes its place so it cannot start."""
    sid = ctx.req["sid"]
    with locked(sid):
        info = read_json(ctx.record_path(sid))
        if info is None:
            if ctx.req.get("record"):
                os.makedirs(ctx.sdir(sid), exist_ok=True)
                ctx.write_info(sid, dict(ctx.req["record"], status="failed", last_deactivated_at=now_iso()))
            return {}
        cancel(sid)
        info.update(status="failed", last_deactivated_at=now_iso())
        ctx.write_info(sid, info)
    return {"info": info}


def op_deactivate(ctx):
    """Cancel the job. An active session's runner writes the final status in its teardown (the client polls for it, then calls with `finish`); a pending one never started a runner — cancelled while queued, or a create that died before submitting — so it is marked inactive right here (a runner that was just starting still writes `inactive` on its way out: it was signalled)."""
    req = ctx.req
    if req.get("finish"):
        sid, _ = ctx.exact()
        with locked(sid):
            info = ctx.info(sid)
            if info.get("status") in LIVE and info.get("job_id") == req["finish"]:
                info.update(status="inactive", last_deactivated_at=now_iso())
                ctx.write_info(sid, info)
        return {"sid": sid, "info": info}
    sid, info, rest, note = ctx.resolve()
    with locked(sid):
        info = ctx.info(sid)
        stopped, dropped = [], []
        if info.get("status") in LIVE:
            # Before the cancel: the runner's teardown keeps this reason instead of blaming the scheduler.
            info["stop_reason"] = "cs deactivate"
            ctx.write_info(sid, info)
            # Named in the reply: cancelling a session that already started its jobs must not kill them silently.
            logdir = ctx.sdir(sid) + "/logs"
            stopped = [dict(command_id=r["command_id"], command=read_cmd("%s/%s.cmd" % (logdir, r["command_id"])) or "") for r in command_rows(logdir, activation_mark(ctx, sid, info)) if r["status"] == "running"]
        # A queued command must not start at some later activation the caller no longer expects.
        dropped = _drop_queued(ctx, sid)
        cancel(sid)
        if info.get("status") == "pending":
            info.update(status="inactive", last_deactivated_at=now_iso())
            ctx.write_info(sid, info)
        if info.get("status") != "active":
            return {"sid": sid, "note": note, "info": info, "stopped": stopped, "dropped": dropped}
    return {"sid": sid, "note": note, "info": info, "job_id": info["job_id"], "stopped": stopped, "dropped": dropped}


def write_lines(path, lines):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as fh:
        fh.write("".join(l + "\n" for l in lines))
    os.replace(tmp, path)


def op_sync_prepare(ctx):
    sid, info, rest, note = ctx.resolve()
    ctx.touch_activity(sid)
    wd = ctx.sdir(sid) + "/workdir"
    os.makedirs(wd, exist_ok=True)
    try:
        with open(ctx.sdir(sid) + "/.sync-manifest") as fh:
            manifest = [l.rstrip("\n") for l in fh if l.strip()]
    except FileNotFoundError:
        manifest = []
    return {"sid": sid, "note": note, "info": info, "workdir": wd, "manifest": manifest}


def op_sync_delete(ctx):
    """Remove paths the previous sync placed that the project no longer has. Never follows a symlink: a link planted from inside the container (`workdir/a -> ~`) must not make a manifest entry `a/x` delete outside the workdir."""
    import shutil
    sid, _ = ctx.exact()
    wd = ctx.sdir(sid) + "/workdir"
    errors = []
    parents = set()
    for rel in ctx.req["paths"]:
        parts = [p for p in rel.split("/") if p and p != "."]
        if not parts or ".." in parts:
            errors.append("%s: invalid manifest entry" % rel)
            continue
        cur = wd
        try:
            for i, p in enumerate(parts[:-1]):
                cur = os.path.join(cur, p)
                st = os.lstat(cur)
                if stat.S_ISLNK(st.st_mode):
                    raise Fail("%s is a symlink — not following it" % "/".join(parts[:i + 1]))
                if not stat.S_ISDIR(st.st_mode):
                    raise FileNotFoundError
            target = os.path.join(cur, parts[-1])
            if stat.S_ISDIR(os.lstat(target).st_mode):
                shutil.rmtree(target)
            else:
                os.unlink(target)
            parents.add(cur)
        except FileNotFoundError:
            continue
        except (OSError, Fail) as exc:
            errors.append("%s: %s" % (rel, exc))
    # Prune directories the deletions emptied (deepest first, never the workdir itself or anything the deletions did not touch).
    for d in sorted(parents, key=len, reverse=True):
        while d != wd and inside(d, wd):
            try:
                os.rmdir(d)
            except OSError:
                break
            d = os.path.dirname(d)
    return {"errors": errors}


def op_sync_finish(ctx):
    sid, _ = ctx.exact()
    write_lines(ctx.sdir(sid) + "/.sync-manifest", ctx.req["files"])
    return {}


OPS = {
    "list": op_list, "show": op_show, "session": op_session, "logs": op_logs, "commands": op_commands, "ls": op_ls,
    "path_check": op_path_check, "create": op_create, "clone": op_clone, "submit": op_submit, "abort": op_abort,
    "deactivate": op_deactivate, "sync_prepare": op_sync_prepare, "sync_delete": op_sync_delete, "sync_finish": op_sync_finish,
    "queue": op_queue, "dequeue": op_dequeue, "copy": op_copy, "delete": op_delete,
}


def main():
    # The client may vanish mid-request (Ctrl-C during a submit); the work must still complete.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    req = json.loads(sys.stdin.buffer.read().decode())
    try:
        ctx = Ctx(req)
        res = OPS[req["op"]](ctx)
        res["user"] = pwd.getpwuid(os.getuid()).pw_name
    except Fail as exc:
        res = dict(exc.extra, error=str(exc), kind=exc.kind)
    except Exception as exc:
        res = {"error": "%s: %s" % (type(exc).__name__, exc), "kind": "remote"}
    sys.stdout.write(req["marker"] + "\n" + json.dumps(res) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
