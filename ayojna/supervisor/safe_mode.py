"""Operator kill switch for L4 safe mode (the same switch as the dashboard button).

    python -m ayojna.supervisor.safe_mode on --reason "storage maintenance"
    python -m ayojna.supervisor.safe_mode off
    python -m ayojna.supervisor.safe_mode status

While it is on, every supervisor cycle still plans and explains, but nothing moves (L4).
"""

from __future__ import annotations

import argparse

from ayojna.api.service import Service
from ayojna.settings import DATA_DIR


def main(a) -> None:
    svc = Service(a.state, DATA_DIR / "lake")
    if a.action != "status":
        flag = svc.set_safe_mode(a.action == "on", a.reason)
        print(f"safe mode {'ON' if flag['on'] else 'OFF'}" + (f": {a.reason}" if flag["on"] else ""))
    st = svc.status()
    if st["l4_reason"]:
        print(f"level now: L4 ({st['l4_reason']})")
    else:
        print(f"last cycle level: {st['level']} (the next cycle runs normally)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["on", "off", "status"])
    ap.add_argument("--reason", default="")
    ap.add_argument("--state", default=str(DATA_DIR / "state"))
    main(ap.parse_args())