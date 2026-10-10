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
# A bare hex tail ("8ec5" for <prefix>_8ec5…) claims a "maybe" token only when a session matches.
HEX_RE = re.compile(r"^[0-9a-f]{4,10}$")
LIVE = ("active", "pending")
# A `.pid` heartbeat (touched every 10 s by a live command) older than this means the command died without recording an exit.
PID_STALE_SECONDS = 60
# `.exit` is written truncate-then-write (twice), and the shared FS shows compute-node writes with a delay — an empty or garbled `.exit` this young is still settling.
EXIT_SETTLE_SECONDS = 10
# squeue states after which the job holds no allocation; COMPLETING included (the runner's final write is guarded against clobbering a re-activated record).
TERMINAL_JOB_STATES = {"COMPLETED", "COMPLETING", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT"}
# Bytes of one log stream per reply (and how far back --tail looks): a bigger log arrives in several requests, so a multi-GB log never sits in the helper's memory, and a stream without line breaks (tqdm bars on stderr) cannot make --tail read it all.
READ_CAP = 8 << 20


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
        raise Fail("absolute paths not allowed: %r" % rel)
    parts = [p for p in rel.split("/") if p and p != "."]
    if ".." in parts:
        raise Fail("parent-directory traversal not allowed: %r" % rel)
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


def job_state(job_id):
    """{state, reason, node, [time_left]} with state ∈ pending/running/gone/unknown. "unknown" means the query itself failed (controller unreachable, timeout) — never to be read as gone."""
    rc, out, err = slurm(["squeue", "-h", "-j", str(job_id), "-o", "%T|%L|%R"], 5)
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
    state, time_left, col = (text.splitlines()[-1].split("|", 2) + ["", ""])[:3]
    if state == "RUNNING":
        return {"state": "running", "reason": None, "node": col, "time_left": time_left or None}
    if state in ("PENDING", "CONFIGURING"):
        reason = col.strip("()")  # squeue prints a pending reason in parentheses, "(None)" when there is none
        return {"state": "pending", "reason": reason if reason not in ("", "None") else None, "node": None, "time_left": time_left or None}
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
            raise Fail("invalid session id: %r" % sid)
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
        """Session selection from the request's `args`: exact id, unique prefix of the id or its hex tail (cwd-project sessions win ties), or no token = the cwd project's only (or only live) session — never "most recent". With `maybe`, the first arg is the session only if it looks like one (commands whose first argument may be a command or a path). Returns (sid, record, rest args, note) — note is set when an explicitly named session belongs to another project."""
        req = self.req
        args = list(req.get("args") or [])
        prefix = req.get("prefix") or ""
        pid = req.get("project_id")
        token = None
        ids = None
        if req.get("maybe"):
            t = args[0].strip().lower() if args else ""
            if t.startswith(prefix + "_"):
                token = t
            elif HEX_RE.match(t):
                ids = self.ids()
                if any(_id_matches(i, t) for i in ids):
                    token = t
            if token is not None:
                args = args[1:]
        elif args:
            token = args[0].strip().lower() or None
            args = args[1:]
        if token is not None:
            if token.startswith(prefix + "_") and SID_RE.match(token):
                rec = read_json(self.record_path(token))
                if rec is not None:
                    return token, rec, args, _note(rec, pid)
            recs = {r["session_id"]: r for r in self.records([i for i in (ids if ids is not None else self.ids()) if _id_matches(i, token)])}
            if len(recs) > 1:
                own = [s for s, r in recs.items() if r.get("project_id") == pid]
                if len(own) == 1:
                    return own[0], recs[own[0]], args, None
                m = sorted(recs)
                raise Fail("session prefix %r is ambiguous: %s" % (token, ", ".join(m[:6]) + (", … +%d more" % (len(m) - 6) if len(m) > 6 else "")))
            if recs:
                sid, rec = next(iter(recs.items()))
                return sid, rec, args, _note(rec, pid)
            raise Fail("no session matches %r — `cs list` shows known sessions" % token)
        mine = [r for r in self.records() if r.get("project_id") == pid]
        if len(mine) == 1:
            return mine[0]["session_id"], mine[0], args, None
        live = [r for r in mine if r.get("status") in LIVE]
        if len(live) == 1:
            return live[0]["session_id"], live[0], args, None
        if not mine:
            raise Fail("no sessions for project %r — create one with `cs create`, or pass a session id" % pid)
        raise Fail("ambiguous session", kind="infer", project_id=pid, sessions=live or mine, live=bool(live))

    def exact(self):
        sid = self.req["sid"]
        return sid, self.info(sid)


def _id_matches(sid, token):
    return sid.startswith(token) or sid.partition("_")[2].startswith(token)


def _note(rec, pid):
    return (rec.get("project_path") or rec.get("project_id")) if rec.get("project_id") != pid else None


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


def cmd_status(base):
    """(status, exit_code) of one command from its log files: exited (`.exit` recorded), running (fresh `.pid` heartbeat), killed (heartbeat went stale without an exit: SIGKILL/OOM/session death), missing."""
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
        return "missing", None
    return ("running" if age <= PID_STALE_SECONDS else "killed"), None


def command_rows(logdir):
    """All recorded commands, newest first: {command_id, status, [exit_code, started_at, exited_at]}. started_at comes from `.start` (written once at launch) — the `.pid` mtime is the heartbeat. Internal `cmd_venvsnap_*` housekeeping is hidden."""
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
        if ext not in ("start", "pid", "exit"):
            continue
        path = os.path.join(logdir, name)
        try:
            mtime = os.lstat(path).st_mtime
        except OSError:
            continue
        row = rows.setdefault(base, {"command_id": base, "status": "running", "exit_code": None, "started_at": None, "exited_at": None, "_pid_age": None, "_exit_age": None})
        if ext == "start":
            row["started_at"] = mtime
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
        pid_age, exit_age = row.pop("_pid_age"), row.pop("_exit_age")
        if exit_age is not None and (row["exit_code"] is not None or exit_age >= EXIT_SETTLE_SECONDS):
            row["status"] = "exited"
        elif exit_age is None and (pid_age is None or pid_age > PID_STALE_SECONDS):
            row["status"] = "killed"
        else:
            row["exited_at"] = None
    ordered = sorted(rows.values(), key=lambda r: r["started_at"] or r["exited_at"] or 0, reverse=True)
    for r in ordered:
        for k in ("started_at", "exited_at"):
            if r[k] is not None:
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
    req = ctx.req
    recs = ctx.records()
    if not req.get("all"):
        # Same-directory sessions under another project id are kept too: the client flags them (the id derives from the git remote and changes when one is added).
        recs = [r for r in recs if r.get("project_id") == req.get("project_id") or r.get("project_path") == req.get("cwd")]
    return {"sessions": recs}


def reconcile(ctx, sid, info):
    """(record, job state). A job that is gone while the record still says pending/active (the runner died without cleanup) is written back as failed/inactive — compare-and-set: a concurrent activate/deactivate owns a record that has moved on."""
    job = job_state(info["job_id"])
    target = {"pending": "failed", "active": "inactive"}.get(info.get("status"))
    if job["state"] == "gone" and target:
        with locked(sid):
            cur = ctx.info(sid)
            if cur.get("job_id") == info["job_id"] and cur.get("status") == info["status"]:
                cur.update(status=target, last_deactivated_at=now_iso())
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


def op_session(ctx):
    """Resolve, optionally mark activity, and return the record plus — when the container is reachable — the socket to it."""
    sid, info, rest, note = ctx.resolve()
    if ctx.req.get("touch"):
        ctx.touch_activity(sid)
    out = {"sid": sid, "note": note, "rest": rest, "info": info}
    _container_target(ctx, sid, info, out)
    return out


def op_logs(ctx):
    """A command's state and — unless `status_only` — its output from `since` (READ_CAP per stream; `more` asks for another request). With `squeue`, a session whose job is gone is reconciled first, so a stream to a dead session stops retrying."""
    import base64
    req = ctx.req
    sid, info, rest, note = ctx.resolve()
    if len(rest) > 1:
        raise Fail("too many arguments — usage: cs logs [SESSION] [COMMAND_ID]", kind="usage")
    if req.get("squeue") and info.get("job_id"):
        info = reconcile(ctx, sid, info)[0]
    logdir = ctx.sdir(sid) + "/logs"
    out = {"sid": sid, "note": note, "session_status": info.get("status")}
    if rest:
        cid = rest[0]
        if not CID_RE.match(cid):
            raise Fail("invalid command id: %r" % cid)
    else:
        # The newest RUNNING command beats the newest command: a short probe launched after a long job must not hijack the default.
        rows = command_rows(logdir)
        if not rows:
            raise Fail("no commands recorded for %s" % sid)
        chosen = next((r for r in rows if r["status"] == "running"), rows[0])
        cid = chosen["command_id"]
        out["chosen"] = chosen["status"] + (" %s" % chosen["exit_code"] if "exit_code" in chosen else "")
    base = "%s/%s" % (logdir, cid)
    status, code = cmd_status(base)
    out.update(cid=cid, status=status)
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


def op_commands(ctx):
    req = ctx.req
    sid, info, rest, note = ctx.resolve()
    rows = command_rows(ctx.sdir(sid) + "/logs")
    limit, offset = max(0, int(req.get("limit") or 0)), max(0, int(req.get("offset") or 0))
    return {"sid": sid, "note": note, "commands": rows[offset:offset + limit] if limit else rows[offset:], "total": len(rows), "offset": offset, "limit": limit or None}


def _workdir_target(ctx, sid, rel):
    """(clean rel, real path) of a workdir path; refuses a path whose real location is outside the workdir — `[ -d ]`, `ls` and rsync follow a symlinked directory component, so a link planted from inside the container (`workdir/evil -> ~/.ssh`) would otherwise turn file ops into arbitrary reads/writes of the cluster user's files."""
    parts = clean_rel(rel)
    wd = os.path.realpath(ctx.sdir(sid) + "/workdir")
    real = os.path.realpath(os.path.join(wd, *parts))
    if not inside(real, wd):
        raise Fail("remote path %r resolves outside the session workdir (symlink) — refusing to operate" % ("/".join(parts) or "."))
    return "/".join(parts), real, wd


def op_ls(ctx):
    sid, info, rest, note = ctx.resolve()
    if len(rest) > 1:
        raise Fail("too many arguments — usage: cs ls [SESSION] [PATH]", kind="usage")
    ctx.touch_activity(sid)
    rel, real, _ = _workdir_target(ctx, sid, rest[0] if rest else "")
    if not os.path.lexists(real):
        raise Fail("path not found in session %s workdir: %r" % (sid, rel or "."))
    import subprocess
    p = subprocess.run(["ls", "-la", "--", real], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise Fail("ls failed for %r: %s" % (rel or ".", p.stderr.decode(errors="replace").strip() or "unknown error"))
    return {"sid": sid, "note": note, "listing": p.stdout.decode(errors="replace")}


def op_path_check(ctx):
    """For upload/download: kind of a workdir path, whether its parent exists, the kind of `<path>/<child>` (where a place-inside upload lands), and the real path to transfer to/from — all symlink-escape checked."""
    sid, _ = ctx.exact()
    rel, real, wd = _workdir_target(ctx, sid, ctx.req.get("path"))
    child = ctx.req.get("child")
    child_kind = None
    if child:
        creal = os.path.realpath(os.path.join(real, child))
        if not inside(creal, wd):
            raise Fail("remote destination %s/%s resolves outside the session workdir (symlink) — refusing to operate" % (rel or ".", child))
        child_kind = kind_of(creal)
    return {"rel": rel, "real": real, "workdir": wd, "kind": kind_of(real), "parent_ok": os.path.isdir(os.path.dirname(real)), "child_kind": child_kind}


def _check_cap(ctx, project_id, cap):
    if cap and cap > 0:
        live = [r for r in ctx.records() if r.get("project_id") == project_id and r.get("status") in LIVE]
        if len(live) >= cap:
            raise Fail("repo %r already has %d active/pending session(s); limit is %d — deactivate one (%s) or raise max_sessions_per_repo in the config" % (project_id, len(live), cap, ", ".join(r["session_id"] for r in live[:4])))


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
    record = dict(ctx.req["record"], project_id=src_info["project_id"], project_path=src_info.get("project_path", ""))
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
    """Start an activation: refuse while the session still has a job (looked up by its name, so one the record lost track of counts too), write the pending record, sbatch, stamp the job id. The lock is held throughout, so an abort/deactivate arriving mid-submit (Ctrl-C during create: this helper keeps running after the client is gone) waits and then sees — and cancels — the right job."""
    req = ctx.req
    sid, info, rest, note = ctx.resolve()
    with locked(sid):
        info = ctx.info(sid)
        jobs = session_jobs(sid)
        if jobs is None:
            raise Fail("cannot tell whether session %s still has a job (the scheduler query failed) — retry in a moment, or `cs deactivate` first" % sid)
        if jobs:
            raise Fail("session %s is already active (job_id=%s); deactivate it first, then activate to restart with new resources." % (sid, jobs[0]))
        if req.get("manifest") is not None:
            write_lines(ctx.sdir(sid) + "/.sync-manifest", req["manifest"])
        prev = info.get("status"), info.get("job_id")
        # The previous activation's runtime state must not show through while pending — nor its job id: show() would find that job gone and fail the fresh session.
        info.update(resources=req["resources"], idle_timeout_minutes=req["idle_timeout_minutes"], image=req["image"], status="pending", job_id=None, node=None, sshd_port=None, gpu=None, last_activity_at=None)
        ctx.write_info(sid, info)  # before sbatch: the runner reads the record as its first act
        r = req["resources"]
        logs = ctx.sdir(sid) + "/logs"
        os.makedirs(logs, exist_ok=True)  # the scheduler opens --output before the runner runs; a record an abort wrote has no logs dir
        flags =["--export=ALL", "--parsable", "--job-name=cs-" + sid, "--partition=" + r["partition"]]
        if req.get("exclude_nodes"):
            flags.append("--exclude=" + ",".join(req["exclude_nodes"]))
        if r.get("gpus"):
            flags.append("--gpus-per-node=%d" % r["gpus"])
        if r.get("gpu_type"):
            flags.append("--constraint=" + r["gpu_type"])
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
        cancel(sid)
        if info.get("status") == "pending":
            info.update(status="inactive", last_deactivated_at=now_iso())
            ctx.write_info(sid, info)
        if info.get("status") != "active":
            return {"sid": sid, "note": note, "info": info}
    return {"sid": sid, "note": note, "info": info, "job_id": info["job_id"]}


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
