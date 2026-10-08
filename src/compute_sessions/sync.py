from __future__ import annotations

import re
import subprocess
from pathlib import Path

from compute_sessions.errors import SyncError


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


def parse_rsync_output(output: str) -> tuple[list[str], list[str], int]:
    """(added files, updated files, bytes moved) from an `--itemize-changes --stats` run."""
    added, updated = _parse_itemize(output)
    return added, updated, _parse_bytes(output)
