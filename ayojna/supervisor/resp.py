"""A tiny Redis client (RESP2 protocol over a socket): no extra package to install.

Only what the supervisor needs: send a command, read the reply, run a Lua script.
Thread-safe (one lock per connection); reconnects once if the connection dropped.
"""

from __future__ import annotations

import socket
import threading
from urllib.parse import urlparse


class RedisError(Exception):
    pass


class Redis:
    def __init__(self, url: str = "redis://localhost:6379/0", timeout_s: float = 2.0):
        u = urlparse(url)
        self.host, self.port = u.hostname or "localhost", u.port or 6379
        self.password = u.password
        self.db = int((u.path or "/0").lstrip("/") or 0)
        self.timeout = timeout_s
        self.sock = None
        self.buf = b""
        self.lock = threading.Lock()

    # ---------- connection ----------
    def _connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.buf = b""
        if self.password:
            self._roundtrip(("AUTH", self.password))
        if self.db:
            self._roundtrip(("SELECT", self.db))

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    # ---------- protocol ----------
    @staticmethod
    def _encode(args) -> bytes:
        out = [b"*%d\r\n" % len(args)]
        for a in args:
            b = a if isinstance(a, bytes) else str(a).encode()
            out.append(b"$%d\r\n%s\r\n" % (len(b), b))
        return b"".join(out)

    def _line(self) -> bytes:
        while b"\r\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("redis closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def _exact(self, n: int) -> bytes:
        while len(self.buf) < n + 2:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("redis closed the connection")
            self.buf += chunk
        data, self.buf = self.buf[:n], self.buf[n + 2 :]
        return data

    def _reply(self):
        line = self._line()
        kind, rest = line[:1], line[1:]
        if kind == b"+":
            return rest.decode()
        if kind == b"-":
            raise RedisError(rest.decode())
        if kind == b":":
            return int(rest)
        if kind == b"$":
            n = int(rest)
            return None if n < 0 else self._exact(n)
        if kind == b"*":
            n = int(rest)
            return None if n < 0 else [self._reply() for _ in range(n)]
        raise RedisError(f"unexpected reply {line[:40]!r}")

    def _roundtrip(self, args):
        self.sock.sendall(self._encode(args))
        return self._reply()

    def execute(self, *args):
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.sock is None:
                        self._connect()
                    return self._roundtrip(args)
                except (OSError, ConnectionError):
                    self.close()
                    if attempt:
                        raise

    # ---------- helpers ----------
    def ping(self) -> bool:
        return self.execute("PING") == "PONG"

    def eval(self, script: str, keys: list, args: list):
        return self.execute("EVAL", script, len(keys), *keys, *args)