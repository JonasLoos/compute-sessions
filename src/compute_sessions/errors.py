from __future__ import annotations


class ComputeSessionsError(Exception):
    pass


class RemoteError(ComputeSessionsError):
    """A command against a source host (ssh/rsync/scheduler) failed or timed out."""
    def __init__(self, message: str, *, command: str | None = None, stderr: str | None = None, exit_code: int | None = None):
        parts = [message]
        if command:
            parts.append(f"command: {command}")
        if exit_code is not None:
            parts.append(f"exit: {exit_code}")
        if stderr:
            parts.append(f"stderr: {stderr.strip()}")
        super().__init__("\n  ".join(parts))
        self.command = command
        self.stderr = stderr
        self.exit_code = exit_code


class SessionError(ComputeSessionsError):
    pass


class SyncError(ComputeSessionsError):
    pass


class ConfigError(ComputeSessionsError):
    pass
