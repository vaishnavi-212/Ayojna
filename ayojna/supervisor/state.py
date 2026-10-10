"""Shared supervisor state: leader lease + fencing token, checkpoints, audit, dead letters.

Two stores, one interface (slide 4: "Shared state (Redis): lease · checkpoints · result
store · audit log · dead-letter queue"):
  StateStore       files in data/state (runs anywhere, one machine)
  RedisStateStore  Redis is the source of truth for coordination; atomic Lua scripts for the
                   lease and fencing, so two supervisors can never both be leader
Both keep human-readable mirrors (lease.json, audit.jsonl, dlq.jsonl) in data/state for the
dashboard. make_state_store("auto") uses Redis when it answers, otherwise files.

Failover is measured: when a new owner takes a lease the old leader did NOT release (it
crashed or hung), the gap since that leader's last heartbeat is recorded (an upper bound on
the time nobody was in charge). A supervisor that stops normally releases its lease, so a
later restart is not counted as a failover.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import pandas as pd

from ayojna.io import atomic_write_text


class StateStore:
    kind = "file"

    def __init__(self, root: str | Path = "data/state", lease_ttl_s: float = 10.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = lease_ttl_s
        self.last_takeover: dict | None = None  # set by acquire() when leadership changed hands

    # ---------- helpers ----------
    def _write(self, name: str, data: dict) -> None:
        atomic_write_text(self.root / name, json.dumps(data))

    def _read(self, name: str) -> dict | None:
        p = self.root / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _append(self, name: str, event: dict) -> dict:
        event = {"ts": time.time(), **event}
        with open(self.root / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
        return event

    def _note_takeover(self, prev: dict | None, owner: str, token: int, now: float) -> None:
        self.last_takeover = None
        if prev and prev.get("owner") != owner and not prev.get("released"):
            gap = now - prev.get("renewed_at", prev.get("expires", now) - self.ttl)
            self.last_takeover = {"from": prev["owner"], "to": owner, "token": token,
                                  "gap_s": round(gap, 2)}  # fmt: skip

    # ---------- leader lease ----------
    def acquire(self, owner: str) -> int | None:
        """Become leader if nobody holds a live lease. Returns the new fencing token, or None."""
        lease = self._read("lease.json")
        now = time.time()
        if lease and lease["owner"] != owner and lease["expires"] > now:
            return None
        token = (lease or {}).get("token", 0) + (0 if lease and lease["owner"] == owner else 1)
        self._note_takeover(lease, owner, token, now)
        self._write("lease.json", {"owner": owner, "token": token, "expires": now + self.ttl,
                                   "renewed_at": now, "store": self.kind})  # fmt: skip
        return token

    def renew(self, owner: str, token: int) -> bool:
        lease = self._read("lease.json")
        if not lease or lease["owner"] != owner or lease["token"] != token:
            return False  # someone else took over: stop giving orders
        now = time.time()
        lease["expires"], lease["renewed_at"] = now + self.ttl, now
        self._write("lease.json", lease)
        return True

    def is_current(self, token: int) -> bool:
        """Fencing check: only the newest leader's commands are accepted."""
        lease = self._read("lease.json")
        return bool(lease) and lease["token"] == token

    def release(self, owner: str, token: int) -> None:
        """Clean shutdown: give the lease up now (a replica may take over at once)."""
        lease = self._read("lease.json")
        if lease and lease["owner"] == owner and lease["token"] == token:
            lease["expires"], lease["released"] = time.time(), True
            self._write("lease.json", lease)

    # ---------- checkpoints ----------
    def run_dir(self, run_id: str) -> Path:
        d = self.root / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_step(self, run_id: str, step: str, output, info: dict) -> None:
        pd.to_pickle(output, self.run_dir(run_id) / f"{step}.pkl")
        done = self._read(f"run-{run_id}.json") or {"steps": {}}
        done["steps"][step] = info
        self._write(f"run-{run_id}.json", done)

    def load_step(self, run_id: str, step: str):
        done = self._read(f"run-{run_id}.json") or {"steps": {}}
        if step not in done["steps"]:
            return None, None
        return pd.read_pickle(self.run_dir(run_id) / f"{step}.pkl"), done["steps"][step]

    def set_active_run(self, run_id: str | None) -> None:
        self._write("active.json", {"run_id": run_id})

    def active_run(self) -> str | None:
        return (self._read("active.json") or {}).get("run_id")

    # ---------- audit + dead-letter queue ----------
    def audit(self, event: dict) -> None:
        self._append("audit.jsonl", event)

    def dead_letter(self, item: dict) -> None:
        """Work that failed for good (a step that needed its fallback, a rolled-back move):
        kept for a person to inspect or replay, instead of silently disappearing."""
        self._append("dlq.jsonl", item)

    def dead_letters(self, limit: int = 50) -> list[dict]:
        p = self.root / "dlq.jsonl"
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(x) for x in lines if x.strip()]

    # ---------- operator safe mode (L4) ----------
    def safe_mode(self) -> dict | None:
        """{"on": true, "reason": ..., "since": ...} while an operator holds the kill switch."""
        flag = self._read("safe_mode.json")
        return flag if flag and flag.get("on") else None


# Lua runs atomically inside Redis: check-and-set of the lease can never interleave.
_ACQUIRE = """
local cur = redis.call('HGETALL', KEYS[1])
local lease = {}
for i = 1, #cur, 2 do lease[cur[i]] = cur[i + 1] end
local now = tonumber(ARGV[2])
if lease.owner and lease.owner ~= ARGV[1] and tonumber(lease.expires) > now then
  return {0, cjson.encode(lease)}
end
local token
if lease.owner == ARGV[1] then token = tonumber(lease.token) else token = redis.call('INCR', KEYS[2]) end
redis.call('HDEL', KEYS[1], 'released')
redis.call('HSET', KEYS[1], 'owner', ARGV[1], 'token', token,
           'expires', tostring(now + tonumber(ARGV[3])), 'renewed_at', ARGV[2])
return {token, cjson.encode(lease)}
"""
_RELEASE = """
if redis.call('HGET', KEYS[1], 'owner') ~= ARGV[1] then return 0 end
if tonumber(redis.call('HGET', KEYS[1], 'token')) ~= tonumber(ARGV[2]) then return 0 end
redis.call('HSET', KEYS[1], 'expires', ARGV[3], 'released', '1')
return 1
"""
_RENEW = """
if redis.call('HGET', KEYS[1], 'owner') ~= ARGV[1] then return 0 end
if tonumber(redis.call('HGET', KEYS[1], 'token')) ~= tonumber(ARGV[2]) then return 0 end
redis.call('HSET', KEYS[1], 'expires', tostring(tonumber(ARGV[3]) + tonumber(ARGV[4])),
           'renewed_at', ARGV[3])
return 1
"""


class RedisStateStore(StateStore):
    """Coordination state in Redis; data-plane files (plans, catalog) stay in `root`."""

    kind = "redis"

    def __init__(self, url: str, root: str | Path = "data/state", lease_ttl_s: float = 10.0,
                 prefix: str = "ayojna", retention_s: int = 86400):  # fmt: skip
        super().__init__(root, lease_ttl_s)
        from ayojna.supervisor.resp import Redis

        self.r = Redis(url)
        self.url = url
        self.prefix = prefix
        self.retention = retention_s
        self.r.ping()  # fail fast: make_state_store falls back to files

    def k(self, *parts: str) -> str:
        return ":".join([self.prefix, *parts])

    def _lease(self) -> dict | None:
        raw = self.r.execute("HGETALL", self.k("lease"))
        if not raw:
            return None
        d = {raw[i].decode(): raw[i + 1].decode() for i in range(0, len(raw), 2)}
        return {"owner": d["owner"], "token": int(d["token"]), "expires": float(d["expires"]),
                "renewed_at": float(d.get("renewed_at", 0)), "store": "redis"}  # fmt: skip

    def acquire(self, owner: str) -> int | None:
        now = time.time()
        token, prev = self.r.eval(_ACQUIRE, [self.k("lease"), self.k("token")],
                                  [owner, repr(now), repr(self.ttl)])  # fmt: skip
        if not token:
            return None
        prev = json.loads(prev) if prev else {}
        prev = prev if isinstance(prev, dict) and prev.get("owner") else None
        if prev:  # Redis hash values arrive as strings
            prev = {"owner": prev["owner"], "expires": float(prev.get("expires", now)),
                    "renewed_at": float(prev.get("renewed_at", now)),
                    "released": bool(prev.get("released"))}  # fmt: skip
        self._note_takeover(prev, owner, int(token), now)
        self._write("lease.json", self._lease())  # mirror for the dashboard
        return int(token)

    def renew(self, owner: str, token: int) -> bool:
        ok = self.r.eval(_RENEW, [self.k("lease")], [owner, token, repr(time.time()), repr(self.ttl)])
        if ok:
            self._write("lease.json", self._lease())
        return bool(ok)

    def is_current(self, token: int) -> bool:
        cur = self.r.execute("GET", self.k("token"))
        return cur is not None and int(cur) == int(token)

    def release(self, owner: str, token: int) -> None:
        if self.r.eval(_RELEASE, [self.k("lease")], [owner, token, repr(time.time())]):
            self._write("lease.json", {**self._lease(), "released": True})

    def save_step(self, run_id: str, step: str, output, info: dict) -> None:
        self.r.execute("SET", self.k("ckpt", run_id, step), pickle.dumps(output), "EX", self.retention)
        self.r.execute("HSET", self.k("run", run_id), step, json.dumps(info))
        self.r.execute("EXPIRE", self.k("run", run_id), self.retention)

    def load_step(self, run_id: str, step: str):
        info = self.r.execute("HGET", self.k("run", run_id), step)
        blob = self.r.execute("GET", self.k("ckpt", run_id, step)) if info is not None else None
        if blob is None:
            return None, None
        return pickle.loads(blob), json.loads(info)

    def set_active_run(self, run_id: str | None) -> None:
        if run_id is None:
            self.r.execute("DEL", self.k("active"))
        else:
            self.r.execute("SET", self.k("active"), run_id)
        self._write("active.json", {"run_id": run_id})

    def active_run(self) -> str | None:
        v = self.r.execute("GET", self.k("active"))
        return v.decode() if v else None

    def _push(self, key: str, event: dict) -> None:
        self.r.execute("RPUSH", self.k(key), json.dumps(event, default=str))
        self.r.execute("LTRIM", self.k(key), -10000, -1)

    def audit(self, event: dict) -> None:
        self._push("audit", self._append("audit.jsonl", event))

    def dead_letter(self, item: dict) -> None:
        self._push("dlq", self._append("dlq.jsonl", item))

    def dead_letters(self, limit: int = 50) -> list[dict]:
        return [json.loads(x) for x in self.r.execute("LRANGE", self.k("dlq"), -limit, -1)]


def make_state_store(kind: str = "auto", root: str | Path = "data/state",
                     lease_ttl_s: float = 10.0, url: str | None = None) -> StateStore:  # fmt: skip
    """kind: file | redis | auto (Redis if it answers, else files). URL: AYOJNA_REDIS_URL."""
    url = url or os.getenv("AYOJNA_REDIS_URL", "redis://localhost:6379/0")
    if kind in ("redis", "auto"):
        try:
            return RedisStateStore(url, root, lease_ttl_s)
        except Exception:
            if kind == "redis":
                raise
    return StateStore(root, lease_ttl_s)