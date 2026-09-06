from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from compute_sessions.errors import SyncError
from compute_sessions.models import SyncResult
from compute_sessions.ssh import HostAccess


_NON_GIT_BLACKLIST = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _git_files(cwd: Path) -> list[str] | None:
    """Return tracked + untracked-not-ignored files (relative POSIX paths), or None if not a git repo.

    Filters out paths that no longer exist on disk — `git ls-files --cached` still lists files deleted from the working tree but not yet removed from the index, and feeding those to rsync's `--files-from` fails with exit 23 (link_stat: No such file or directory) and wedges the whole sync.
    """
    res = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True, errors="replace",
    )
    if res.returncode != 0 or res.stdout.strip() != "true":
        return None
    res = subprocess.run(
        ["git", "-C", str(cwd), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True, text=True, errors="replace",
    )
    if res.returncode != 0:
        raise SyncError(f"git ls-files failed: {res.stderr.strip()}")
    return [p for p in res.stdout.split("\0") if p and (cwd / p).is_file()]


def _walk_files(cwd: Path) -> list[str]:
    out: list[str] = []
    for p in cwd.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(cwd)
        if any(part in _NON_GIT_BLACKLIST for part in rel.parts):
            continue
        out.append(rel.as_posix())
    return out


def plan_sync(cwd: Path) -> list[str]:
    """Return the ordered list of file paths (POSIX, relative to cwd) to ship to the cluster."""
    files = _git_files(cwd)
    if files is None:
        files = _walk_files(cwd)
    return sorted(set(files))


def _read_manifest(access: HostAccess, manifest_path: str) -> set[str]:
    res = access.run(f"cat {shlex.quote(manifest_path)} 2>/dev/null || true", check=True)
    return {line for line in res.stdout.splitlines() if line}


def _write_manifest(access: HostAccess, manifest_path: str, files: list[str]) -> None:
    body = "\n".join(files) + ("\n" if files else "")
    access.run(f"cat > {shlex.quote(manifest_path)}", check=True, input_text=body)


def _delete_paths(access: HostAccess, workdir: str, rel_paths: list[str]) -> list[str]:
    """Remove each manifest-listed path from workdir; return per-path error messages.

    Uses `rm -rf` (not `rm -f`) so an entry whose local type changed since the last sync — e.g. a file replaced by a directory — still unwinds cleanly. Manifest entries are paths this session's own sync placed, bounded to workdir, so recursive delete stays in the intended scope.

    No `set -e`: one failing path must not wedge the rest of the deletions or the manifest rewrite that follows. Each path's rm stderr is tagged and returned to the caller as warnings.
    """
    if not rel_paths:
        return []
    quoted_wd = shlex.quote(workdir)
    # Trailing newline is mandatory: `while IFS= read -r f` returns non-zero on a final line with no newline, so the loop body skips that last entry (or the only entry, when there's a single path). `_write_manifest` adds it for the same reason.
    lines = "\n".join(rel_paths) + "\n"
    script = (
        f"cd {quoted_wd}\n"
        f"while IFS= read -r f; do\n"
        f"  [ -z \"$f\" ] && continue\n"
        f"  err=$(rm -rf -- \"$f\" 2>&1) || printf 'ERR %s: %s\\n' \"$f\" \"$err\" >&2\n"
        f"done\n"
        # prune empty dirs, deepest first; ignore errors
        f"find . -mindepth 1 -type d -empty -delete 2>/dev/null || true\n"
    )
    res = access.run(script, input_text=lines, check=False)
    return [line[4:] for line in res.stderr.splitlines() if line.startswith("ERR ")]


_ITEMIZE_RE = re.compile(r"^([<>ch.*][fdLDS])(\S*)\s+(.+)$")
_BYTES_RE = re.compile(r"^Total bytes sent:\s+([0-9,]+)", re.MULTILINE)
_BYTES_RECV_RE = re.compile(r"^Total bytes received:\s+([0-9,]+)", re.MULTILINE)


def _parse_itemize(output: str) -> tuple[list[str], list[str]]:
    # rsync itemize format is "YXcstpoguax path" — group(1) captures YX (direction + type, always 2 chars) and group(2) captures the 9-char change-attributes string. A freshly-created file has all-`+` attributes (e.g. "<f++++++++++"), so check group(2), not group(1). The direction char is `<` for a push (data flowing to the remote receiver, which is always our case here) and `>` for a pull/local copy — so match on the type char (`f` = regular file), not a hard-coded direction. Only regular files are reported; dir/symlink itemize lines (`d`/`L`) are ignored.
    added, updated = [], []
    for line in output.splitlines():
        m = _ITEMIZE_RE.match(line)
        if not m:
            continue
        code, flags, name = m.group(1), m.group(2), m.group(3)
        if code[1] != "f":
            continue
        if flags.startswith("+"):
            added.append(name)
        else:
            updated.append(name)
    return added, updated


def _parse_bytes(output: str) -> int:
    total = 0
    for rx in (_BYTES_RE, _BYTES_RECV_RE):
        m = rx.search(output)
        if m:
            total += int(m.group(1).replace(",", ""))
    return total


def sync_session(
    access: HostAccess,
    session_id: str,
    cwd: Path,
    *,
    rebuild_manifest: bool = False,
    follow_symlinks: bool = False,
) -> SyncResult:
    """Mirror `cwd` into the session's workdir and update the manifest.

    When `rebuild_manifest=True`, skip the diff/delete step and just rewrite the manifest to match the current plan. Escape hatch for sessions whose manifest has drifted from workdir state (e.g. out-of-band edits on the cluster) — after this call the next normal sync computes deletions from the fresh baseline instead of the stale one.

    `follow_symlinks=True` dereferences symlinks during rsync so the linked file contents are copied (rsync `-L`). Default preserves links as-is.
    """
    new_files = plan_sync(cwd)
    workdir = access.paths.workdir(session_id)
    manifest = access.paths.manifest(session_id)

    # Ensure workdir exists.
    access.run(f"mkdir -p {shlex.quote(workdir)}")

    # Remove what the previous sync placed and the project no longer has — BEFORE the transfer: a path whose type changed locally (a `foo/` directory replaced by a file `foo`, or the reverse) must be unwound first, since rsync refuses to write a file over a non-empty directory. If rsync then fails, the manifest keeps the old entries and the next sync repeats the (idempotent) deletions.
    cleanup_errors: list[str] = []
    deleted_outdated: list[str] = []
    if not rebuild_manifest:
        old_manifest = _read_manifest(access, manifest)
        new_set = set(new_files)
        deleted_outdated = sorted(old_manifest - new_set)
        cleanup_errors = _delete_paths(access, workdir, deleted_outdated)

    # rsync new set into workdir.
    output = (
        access.rsync_files(cwd, new_files, workdir, follow_symlinks=follow_symlinks)
        if new_files else ""
    )
    added, updated = _parse_itemize(output)
    bytes_transferred = _parse_bytes(output)

    # Always rewrite the manifest — even on cleanup errors. Leaving the old manifest behind after a partial cleanup would wedge every subsequent sync on the same failing entries.
    _write_manifest(access, manifest, new_files)

    return SyncResult(
        added=sorted(added),
        updated=sorted(updated),
        deleted_outdated=deleted_outdated,
        bytes_transferred=bytes_transferred,
        cleanup_errors=cleanup_errors,
    )
