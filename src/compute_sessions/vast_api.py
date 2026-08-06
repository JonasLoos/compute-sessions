from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from compute_sessions.config import SourceConfig
from compute_sessions.errors import RemoteError, SessionError


# Deliberately a hand-rolled stdlib client: the official `vastai` package pins ~17 heavy dependencies (pillow, cryptography, aiohttp, a PDF library, ...) for what is, for us, six endpoints.
API_BASE = "https://console.vast.ai/api/v0"
_TIMEOUT = 30.0


class VastClient:
    """Minimal vast.ai REST client for the instance lifecycle: search offers, rent, poll, destroy, fetch logs."""

    def __init__(self, source: SourceConfig):
        self.source = source
        self._key: str | None = None

    def _api_key(self) -> str:
        if self._key is None:
            p = Path(self.source.api_key_file).expanduser()
            if not p.is_file():
                raise SessionError(
                    f"vast API key file not found at {p} — put your vast.ai API key there "
                    f"(console.vast.ai → Keys), or point `api_key_file` in the source config at it"
                )
            self._key = p.read_text().strip()
            if not self._key:
                raise SessionError(f"vast API key file {p} is empty")
        return self._key

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RemoteError(
                f"vast API {method} {path} failed",
                command=f"{method} {API_BASE}{path}",
                stderr=f"HTTP {exc.code}: {detail}",
                exit_code=exc.code,
            ) from exc
        except (OSError, ValueError) as exc:
            raise RemoteError(
                f"vast API {method} {path} failed",
                command=f"{method} {API_BASE}{path}",
                stderr=str(exc),
                exit_code=None,
            ) from exc

    def search_offers(self, filters: dict, limit: int = 10) -> list[dict]:
        """On-demand offers matching `filters` ({field: {op: value}}), cheapest first."""
        body = dict(filters)
        body["type"] = "on-demand"
        body["order"] = [["dph_total", "asc"]]
        body["limit"] = limit
        return self._request("POST", "/bundles/", body).get("offers", []) or []

    def create_instance(self, offer_id: int, body: dict) -> str:
        res = self._request("PUT", f"/asks/{offer_id}/", body)
        if not res.get("success") or not res.get("new_contract"):
            raise SessionError(f"vast instance creation failed: {res}")
        return str(res["new_contract"])

    def show_instance(self, instance_id: str) -> dict | None:
        """Instance payload, or None when vast has no such instance (destroyed)."""
        try:
            res = self._request("GET", f"/instances/{instance_id}/?owner=me")
        except RemoteError as exc:
            if exc.exit_code == 404:
                return None
            raise
        inst = res.get("instances")
        return inst if isinstance(inst, dict) and inst else None

    def list_instances(self) -> list[dict]:
        return self._request("GET", "/instances/?owner=me").get("instances", []) or []

    def destroy_instance(self, instance_id: str) -> None:
        """Destroy = the ONLY full billing cutoff (stopped/exited instances keep billing storage). 404 counts as success — already gone."""
        try:
            self._request("DELETE", f"/instances/{instance_id}/", {})
        except RemoteError as exc:
            if exc.exit_code != 404:
                raise

    def instance_logs(self, instance_id: str, tail: int = 200) -> str:
        """Container (docker) logs via vast's async relay: request → poll the returned S3 URL. Empty string when unavailable (host offline, instance destroyed)."""
        try:
            res = self._request("PUT", f"/instances/request_logs/{instance_id}/", {"tail": str(tail)})
        except RemoteError:
            return ""
        url = res.get("result_url")
        if not url:
            return ""
        for _ in range(20):
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    if resp.status == 200:
                        return resp.read().decode(errors="replace")
            except (urllib.error.HTTPError, OSError):
                pass
            time.sleep(0.4)
        return ""


_FILTER_OPS = (("<=", "lte"), (">=", "gte"), ("!=", "neq"), ("=", "eq"), ("<", "lt"), (">", "gt"))


def _coerce(raw: str) -> bool | float | str:
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_filter_clauses(text: str) -> dict:
    """Parse a config offer-filter string ("verified=true reliability>0.98 inet_down>=500") into vast /bundles/ filter objects ({field: {op: value}})."""
    out: dict = {}
    for clause in text.split():
        for sym, op in _FILTER_OPS:
            if sym in clause:
                field, _, raw = clause.partition(sym)
                if not field or not raw:
                    raise SessionError(f"invalid offer_filter clause {clause!r}")
                out.setdefault(field, {})[op] = _coerce(raw)
                break
        else:
            raise SessionError(f"invalid offer_filter clause {clause!r} (expected <field><op><value> with op in =, !=, <, >, <=, >=)")
    return out


class Ledger:
    """Append-only JSONL of instance start/stop events for one vast source — the basis of the monthly max_total_spend estimate.

    Instance ids are unique contract ids, so each gets exactly one start and at most one stop. Instances that disappear without a stop (runner self-destructed on idle, or destroyed via the console) are closed out by reconcile(); their synthesized stop time is capped at start + max_session_hours — the runner enforces that bound on-instance — so the estimate errs high, boundedly.
    """

    def __init__(self, path: Path):
        self.path = path

    def _events(self) -> list[dict]:
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return []
        out = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
        return out

    def _append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def start(self, instance_id: str, session_id: str, dph: float) -> None:
        self._append({"ts": time.time(), "event": "start", "instance_id": str(instance_id), "session_id": session_id, "dph": dph})

    def stop(self, instance_id: str, session_id: str, ts: float | None = None) -> None:
        self._append({"ts": ts if ts is not None else time.time(), "event": "stop", "instance_id": str(instance_id), "session_id": session_id})

    def open_instances(self) -> dict[str, dict]:
        """instance_id → start event, for starts with no stop yet."""
        open_: dict[str, dict] = {}
        for ev in self._events():
            iid = str(ev.get("instance_id", ""))
            if ev.get("event") == "start":
                open_[iid] = ev
            elif ev.get("event") == "stop":
                open_.pop(iid, None)
        return open_

    def close(self, instance_id: str, max_session_hours: int) -> None:
        """Synthesize a stop for an instance observed to be gone, capped at its enforced max lifetime."""
        ev = self.open_instances().get(str(instance_id))
        if ev is None:
            return
        ts = min(time.time(), float(ev.get("ts", 0)) + max_session_hours * 3600)
        self.stop(instance_id, ev.get("session_id", ""), ts=ts)

    def reconcile(self, live_ids: set[str], max_session_hours: int) -> None:
        """Close out every ledger-open instance that no longer exists on vast."""
        for iid in self.open_instances():
            if iid not in live_ids:
                self.close(iid, max_session_hours)

    def month_spend(self) -> float:
        """$ accrued in the current calendar month (UTC): each start→stop interval (→now for open instances) clipped to the month, times the instance's $/hr rate."""
        now = time.time()
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        stops: dict[str, float] = {}
        events = self._events()
        for ev in events:
            if ev.get("event") == "stop":
                stops[str(ev.get("instance_id", ""))] = float(ev.get("ts", now))
        total = 0.0
        for ev in events:
            if ev.get("event") != "start":
                continue
            start = float(ev.get("ts", now))
            end = stops.get(str(ev.get("instance_id", "")), now)
            lo = max(start, month_start)
            hi = min(end, now)
            if hi > lo:
                total += (hi - lo) / 3600 * float(ev.get("dph") or 0)
        return total
