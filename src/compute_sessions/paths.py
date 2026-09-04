from __future__ import annotations

import re

from compute_sessions.errors import SessionError

_SESSION_ID_RE = re.compile(r"^[a-z0-9_]+$")
# Explicit ASCII allowlist rather than str.isalnum(), which also accepts Unicode letters and digits (e.g. "café"). Every downstream use is shlex.quote'd and "." / "/" are excluded so traversal was never possible, but an ASCII-only id avoids any future foot-gun.
_COMMAND_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.fullmatch(session_id) or len(session_id) > 64:
        raise SessionError(f"invalid session id: {session_id!r}")


def validate_command_id(command_id: str) -> None:
    if not _COMMAND_ID_RE.fullmatch(command_id) or len(command_id) > 128:
        raise SessionError(f"invalid command_id: {command_id!r}")


class RemotePaths:
    """Resolves POSIX paths on the cluster relative to the resolved `remote_base`.

    All paths are returned as strings suitable for interpolation into `ssh` commands.
    """

    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def sessions_root(self) -> str:
        return f"{self.base}/sessions"

    def runner(self) -> str:
        return f"{self.base}/runner.py"

    def session_dir(self, session_id: str) -> str:
        validate_session_id(session_id)
        return f"{self.base}/sessions/{session_id}"

    def config_file(self, session_id: str) -> str:
        return f"{self.session_dir(session_id)}/config.json"

    def workdir(self, session_id: str) -> str:
        return f"{self.session_dir(session_id)}/workdir"

    def logs_dir(self, session_id: str) -> str:
        return f"{self.session_dir(session_id)}/logs"

    def socket(self, session_id: str) -> str:
        return f"{self.session_dir(session_id)}/session.sock"

    def manifest(self, session_id: str) -> str:
        return f"{self.session_dir(session_id)}/.sync-manifest"

    @staticmethod
    def _clean_parts(rel: str) -> list[str]:
        """Split a user-supplied relative path into safe components; rejects `..` and absolute paths so operations stay scoped to their root."""
        if rel.startswith("/"):
            raise SessionError(f"absolute paths not allowed: {rel!r}")
        parts = [p for p in rel.split("/") if p and p != "."]
        for p in parts:
            if p == "..":
                raise SessionError(f"parent-directory traversal not allowed: {rel!r}")
        return parts

    def resolve_workdir_relative(self, session_id: str, rel: str) -> str:
        """Resolve a user-supplied path relative to the session's workdir."""
        parts = self._clean_parts((rel or "").strip())
        wd = self.workdir(session_id)
        return f"{wd}/" + "/".join(parts) if parts else wd
