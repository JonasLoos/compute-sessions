from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    return _SLUG_RE.sub("-", s.lower()).strip("-") or "unnamed"


def _parse_remote(url: str) -> str | None:
    """Extract 'owner/repo' (or similar) from a git remote URL."""
    s = url.strip()
    if not s:
        return None
    # strip trailing .git
    if s.endswith(".git"):
        s = s[:-4]
    # git@host:owner/repo
    if "@" in s and ":" in s and not s.startswith(("http://", "https://", "ssh://")):
        _, _, path = s.partition(":")
    elif "://" in s:
        # scheme://[user@]host/path
        _, _, rest = s.partition("://")
        _, _, path = rest.partition("/")
    else:
        path = s
    path = path.strip("/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        return None
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def project_id(cwd: Path | str | None = None) -> str:
    """Derive a stable project identifier for the current working directory.

    Prefers the `origin` git remote; falls back to directory basename + a short hash of the absolute path so two local clones don't collide.
    """
    p = Path(cwd) if cwd else Path.cwd()
    try:
        res = subprocess.run(
            ["git", "-C", str(p), "remote", "get-url", "origin"],
            capture_output=True, text=True, check=False,
        )
        if res.returncode == 0:
            parsed = _parse_remote(res.stdout)
            if parsed:
                return _slug(parsed.replace("/", "-"))
    except FileNotFoundError:
        pass

    resolved = p.resolve()
    base = _slug(resolved.name)
    digest = hashlib.sha1(str(resolved).encode()).hexdigest()[:8]
    return f"{base}-{digest}"
