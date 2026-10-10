"""Ingest the Alibaba Cloud block traces (2020) into the same extent_hourly table as MSR.

Format (github.com/alibaba/block-traces, file io_traces.csv, one I/O per row):
    device_id, opcode (R|W), offset (bytes), length (bytes), timestamp (microseconds)
The full trace is 181 GB compressed / 751 GB raw for 1,000 virtual disks over a month, so
this reader STREAMS it and stops early: it keeps only the devices you ask for and only the
first `--days` days. It reads io_traces.csv, .csv.gz, the original .tar.gz, or an https
link straight to the file (nothing has to be downloaded in full).

    python -m ayojna.ingest.alibaba --src alibaba_block_traces_2020.tar.gz --first 8 --days 5
    python -m ayojna.ingest.alibaba --src io_traces.csv --devices 3 17 42 --days 5

Same maths as the MSR ingest: 256 MB extents, hourly reads/writes/bytes, sequential share.
Volumes are named ali_<device_id> (compliance tags: the defaults, general SLA).
"""

from __future__ import annotations

import argparse
import gzip
import io
import tarfile
import time
import urllib.request
from contextlib import contextmanager

import numpy as np
import pandas as pd

from ayojna.contracts import EXTENT_BYTES, EXTENT_HOURLY_COLUMNS, validate_frame
from ayojna.io import write_table

COLS = ["device_id", "opcode", "offset", "length", "timestamp"]
US_PER_HOUR = 3_600_000_000
CHUNK = 2_000_000


class _Unseekable(io.RawIOBase):
    """A file inside a streamed tar archive, wrapped so pandas can read it chunk by chunk."""

    def __init__(self, f):
        self.f = f

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        data = self.f.read(len(buf))
        buf[: len(data)] = data
        return len(data)


@contextmanager
def _open(src: str):
    """A text stream of io_traces.csv from a path, a .gz, a .tar.gz or an https URL."""
    raw = urllib.request.urlopen(src) if src.startswith(("http://", "https://")) else open(src, "rb")
    try:
        name = src.lower().split("?")[0]
        if name.endswith((".tar.gz", ".tgz", ".tar")):
            tar = tarfile.open(fileobj=raw, mode="r|*")  # streaming: no seeking, no full download
            for member in tar:
                if member.name.endswith("io_traces.csv"):
                    yield io.BufferedReader(_Unseekable(tar.extractfile(member)), 1 << 20)
                    return
            raise FileNotFoundError("io_traces.csv not found in the archive")
        if name.endswith(".gz"):
            yield gzip.open(raw)
            return
        yield raw
    finally:
        raw.close()


def _clean(chunk: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({c: pd.to_numeric(chunk[c], errors="coerce")
                       for c in ("device_id", "offset", "length", "timestamp")})  # fmt: skip
    op = chunk["opcode"].astype(str).str.strip().str.upper()
    ok = df.notna().all(axis=1) & op.isin(["R", "W"]) & (df["length"] > 0) & (df["offset"] >= 0)
    df = df[ok].astype("int64")
    df["is_read"] = (op[ok] == "R").to_numpy()
    return df


def build_alibaba(src: str, out: str, devices: list[int] | None = None, first: int = 8,
                  days: float = 5.0, chunksize: int = CHUNK) -> pd.DataFrame:  # fmt: skip
    t0 = time.time()
    keep: set[int] | None = set(devices) if devices else None
    start = stop = None
    last_end: dict[int, int] = {}  # per device: end of its previous I/O (sequential detection)
    parts, rows_read = [], 0
    with _open(src) as f:
        for chunk in pd.read_csv(f, header=None, names=COLS, chunksize=chunksize, dtype=str):
            df = _clean(chunk)
            rows_read += len(chunk)
            if df.empty:
                continue
            if start is None:
                start = int(df["timestamp"].min())
                stop = start + int(days * 24 * US_PER_HOUR)
                if keep is None:  # the first N devices that appear in the trace
                    keep = set(pd.unique(df["device_id"])[:first].tolist())
                print(f"devices {sorted(keep)}, first {days:g} days")
            if df["timestamp"].min() >= stop:
                break  # the trace is time-ordered: everything after this is out of the window
            df = df[df["device_id"].isin(keep) & (df["timestamp"] < stop)]
            if df.empty:
                continue
            df = df.sort_values(["device_id", "timestamp"], kind="stable")
            end = df["offset"] + df["length"]
            prev = end.groupby(df["device_id"]).shift(1)
            firsts = prev.isna()
            prev[firsts] = df.loc[firsts, "device_id"].map(last_end).astype(float)
            last_end.update(end.groupby(df["device_id"]).last().to_dict())
            df = df.assign(
                hour=(df["timestamp"] - start) // US_PER_HOUR,
                extent_id=df["offset"] // EXTENT_BYTES,
                seq=(df["offset"] == prev),
                read_bytes=np.where(df["is_read"], df["length"], 0),
                write_bytes=np.where(df["is_read"], 0, df["length"]),
            )
            parts.append(df.groupby(["device_id", "extent_id", "hour"]).agg(
                reads=("is_read", "sum"), ios=("is_read", "size"), read_bytes=("read_bytes", "sum"),
                write_bytes=("write_bytes", "sum"), size_sum=("length", "sum"), seq=("seq", "sum"),
            ))  # fmt: skip
            print(f"  {rows_read:,} rows read, {sum(len(p) for p in parts):,} extent-hours kept",
                  end="\r")  # fmt: skip
    if not parts:
        raise ValueError("no I/O for the chosen devices in the chosen window")
    g = pd.concat(parts).groupby(level=["device_id", "extent_id", "hour"]).sum().reset_index()
    g["volume"] = "ali_" + g["device_id"].astype(str)
    g["writes"] = g["ios"] - g["reads"]
    g["avg_io_size"] = g["size_sum"] / g["ios"]
    g["rand_ratio"] = 1.0 - g["seq"] / g["ios"]
    table = validate_frame(g, EXTENT_HOURLY_COLUMNS, "extent_hourly")
    # no common-window cut: one trace records every device for the whole period, so a quiet
    # device is genuinely idle (unlike MSR, where separate traces end at different times)
    table = table.sort_values(["volume", "extent_id", "hour"]).reset_index(drop=True)
    write_table(table, out)
    print(f"\nalibaba extent_hourly: {len(table):,} rows, {table['volume'].nunique()} volumes, "
          f"hours 0-{int(table['hour'].max())} ({time.time() - t0:.0f}s) -> {out}")  # fmt: skip
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Alibaba block traces -> extent_hourly")
    ap.add_argument("--src", required=True, help="io_traces.csv[.gz], the .tar.gz, or an https link")
    ap.add_argument("--devices", nargs="*", type=int, help="device ids (default: the first --first)")
    ap.add_argument("--first", type=int, default=8)
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--out", default="data/lake/alibaba_eh.parquet")
    a = ap.parse_args()
    build_alibaba(a.src, a.out, a.devices, a.first, a.days)