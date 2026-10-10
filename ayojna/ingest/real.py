"""Ingest the REAL MSR Cambridge traces, straight from the downloaded .tar files.

No need to unzip anything: put the SNIA download(s) (.tar / .tgz / .zip, or extracted
.csv / .csv.gz files, in any sub-folder) in one folder and run

    python -m ayojna.ingest.real --raw data/raw/msr --list          # what is inside
    python -m ayojna.ingest.real --raw data/raw/msr                 # build extent_hourly

Files are read in chunks (streaming), so a 5 GB download never has to fit in memory.
The result is identical to ingest/msr.py (same columns, same 256 MB extents, one clock).
"""

from __future__ import annotations

import argparse
import gzip
import re
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ayojna.contracts import EXTENT_BYTES, EXTENT_HOURLY_COLUMNS, validate_frame
from ayojna.ingest.msr import MAX_REJECT_SHARE, MSR_COLUMNS, TICKS_PER_HOUR, IngestError
from ayojna.io import write_table

# Volumes that carry compliance tags in config/compliance.yaml (legal hold, pii, financial...)
DEFAULT_VOLUMES = ["web_0", "src1_2", "usr_0", "ts_0", "hm_0", "prn_0", "proj_0", "mds_0"]
NAME = re.compile(r"^([a-z]+\d*_\d+)\.csv(\.gz)?$", re.I)
CHUNK = 1_000_000


@dataclass
class Source:
    volume: str
    path: Path  # a .csv/.csv.gz file, or the archive that contains it
    member: str | None = None  # name inside the archive


def _is_tar(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith((".tar", ".tgz", ".tar.gz"))


def discover(raw_dir: str | Path) -> dict[str, Source]:
    """Every MSR volume under raw_dir: loose .csv/.csv.gz files and members of .tar/.zip files."""
    found: dict[str, Source] = {}

    def add(base: str, src: Source) -> None:
        m = NAME.match(base)
        if m:
            src.volume = m.group(1).lower()
            found.setdefault(src.volume, src)

    for p in sorted(Path(raw_dir).rglob("*")):
        if not p.is_file():
            continue
        if _is_tar(p):
            with tarfile.open(p, "r:*") as tar:
                for m in tar.getmembers():
                    if m.isfile():
                        add(Path(m.name).name, Source("", p, m.name))
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if not name.endswith("/"):
                        add(Path(name).name, Source("", p, name))
        else:
            add(p.name, Source("", p))
    return found


def _chunks(src: Source, chunksize: int):
    """Yield raw CSV chunks of one volume; closes every file handle when done."""
    archive = None
    if src.member is None:
        raw = open(src.path, "rb")
    elif _is_tar(src.path):
        archive = tarfile.open(src.path, "r:*")
        raw = archive.extractfile(src.member)
    else:
        archive = zipfile.ZipFile(src.path)
        raw = archive.open(src.member)
    name = src.member or src.path.name
    stream = gzip.open(raw) if name.lower().endswith(".gz") else raw
    try:
        yield from pd.read_csv(
            stream,
            header=None,
            names=MSR_COLUMNS,
            usecols=["Timestamp", "Type", "Offset", "Size"],
            dtype=str,
            chunksize=chunksize,
        )
    finally:
        stream.close()
        raw.close()
        if archive is not None:
            archive.close()


def _clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ts = pd.to_numeric(raw["Timestamp"], errors="coerce")
    offset = pd.to_numeric(raw["Offset"], errors="coerce")
    size = pd.to_numeric(raw["Size"], errors="coerce")
    kind = raw["Type"].astype(str).str.strip().str.lower()
    ok = ts.notna() & offset.notna() & size.notna() & (offset >= 0) & (size > 0)
    ok &= kind.isin(["read", "write"])
    df = pd.DataFrame(
        {
            "ts": ts[ok].astype("int64"),
            "is_read": kind[ok] == "read",
            "offset": offset[ok].astype("int64"),
            "size": size[ok].astype("int64"),
        }
    )
    df = df.drop_duplicates(subset=["ts", "offset", "size", "is_read"])
    return df.sort_values("ts", kind="stable"), int(len(raw) - ok.sum())


def first_timestamp(src: Source, rows: int = 100_000) -> int:
    """MSR files are time-ordered, so the start is in the first rows."""
    gen = _chunks(src, rows)
    try:
        first = next(gen)
    finally:
        gen.close()
    return int(pd.to_numeric(first["Timestamp"], errors="coerce").min())


def volume_extent_hourly(src: Source, start_ts: int, chunksize: int = CHUNK) -> pd.DataFrame:
    """Stream one volume into extent_hourly rows (same maths as msr.to_extent_hourly)."""
    parts, prev_end, n_raw, rejected = [], np.nan, 0, 0
    for raw in _chunks(src, chunksize):
        df, bad = _clean(raw)
        n_raw, rejected = n_raw + len(raw), rejected + bad
        if df.empty:
            continue
        if (df["ts"] < start_ts).any():
            raise IngestError(f"{src.volume}: events before the start time (file not ordered?)")
        end = df["offset"] + df["size"]
        prev = end.shift(1)
        prev.iloc[0] = prev_end  # carry the last I/O of the previous chunk
        prev_end = float(end.iloc[-1])
        df = df.assign(
            hour=(df["ts"] - start_ts) // TICKS_PER_HOUR,
            extent_id=df["offset"] // EXTENT_BYTES,
            seq=(df["offset"] == prev),
            read_bytes=np.where(df["is_read"], df["size"], 0),
            write_bytes=np.where(df["is_read"], 0, df["size"]),
        )
        parts.append(
            df.groupby(["extent_id", "hour"]).agg(
                reads=("is_read", "sum"),
                ios=("is_read", "size"),
                read_bytes=("read_bytes", "sum"),
                write_bytes=("write_bytes", "sum"),
                size_sum=("size", "sum"),
                seq=("seq", "sum"),
            )
        )
    if n_raw == 0:
        raise IngestError(f"{src.volume}: empty file")
    if rejected / n_raw > MAX_REJECT_SHARE:
        raise IngestError(f"{src.volume}: {rejected}/{n_raw} rows rejected, file looks broken")
    g = pd.concat(parts).groupby(level=["extent_id", "hour"]).sum().reset_index()
    g["volume"] = src.volume
    g["writes"] = g["ios"] - g["reads"]
    g["avg_io_size"] = g["size_sum"] / g["ios"]
    g["rand_ratio"] = 1.0 - g["seq"] / g["ios"]
    return validate_frame(g, EXTENT_HOURLY_COLUMNS, "extent_hourly")


def common_window(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only the hours where EVERY volume was being traced.

    Traces do not all stop at the same time (MSR: ts_0 runs ~26 h longer than the rest).
    After a trace stops, its volume would look idle, which flatters every tiering policy
    and mislabels its extents as cold. So the replay ends where the shortest trace ends.
    """
    end = int(table.groupby("volume")["hour"].max().min())
    dropped = int((table["hour"] > end).sum())
    if dropped:
        print(f"common window: hours 0-{end} ({dropped:,} rows after a trace ended removed)")
    return table[table["hour"] <= end]


def build_real(
    raw_dir: str | Path,
    out_path: str | Path,
    volumes: list[str] | None = None,
    chunksize: int = CHUNK,
) -> pd.DataFrame:
    found = discover(raw_dir)
    if not found:
        raise FileNotFoundError(f"no MSR .tar / .zip / .csv / .csv.gz files in {raw_dir}")
    wanted = volumes or DEFAULT_VOLUMES
    if wanted == ["all"]:
        wanted = sorted(found)
    missing = [v for v in wanted if v not in found]
    use = [found[v] for v in wanted if v in found]
    if missing:
        print(f"not in your download (skipped): {', '.join(missing)}")
    if not use:
        raise FileNotFoundError(f"none of {wanted} found; available: {', '.join(sorted(found))}")
    start = min(first_timestamp(s) for s in use)  # one clock for every volume
    tables = []
    for s in use:
        t0 = time.time()
        t = volume_extent_hourly(s, start, chunksize)
        print(
            f"{s.volume:<8} {int((t['reads'] + t['writes']).sum()):>12,} I/Os  "
            f"{t['extent_id'].nunique():>6,} active extents  ({time.time() - t0:.0f}s)"
        )
        tables.append(t)
    table = pd.concat(tables, ignore_index=True).sort_values(["volume", "extent_id", "hour"])
    table = common_window(table).reset_index(drop=True)
    write_table(table, out_path)
    print(
        f"\nextent_hourly: {len(table):,} rows, {table['volume'].nunique()} volumes, "
        f"hours 0-{int(table['hour'].max())} -> {out_path}"
    )
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real MSR Cambridge traces -> extent_hourly")
    ap.add_argument("--raw", default="data/raw/msr")
    ap.add_argument("--out", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--volumes", nargs="*", help="e.g. web_0 usr_0, or: all (default: 8 tagged)")
    ap.add_argument("--list", action="store_true", help="only list the volumes found")
    a = ap.parse_args()
    if a.list:
        for v, s in sorted(discover(a.raw).items()):
            print(f"{v:<8} {s.path.name}{' :: ' + s.member if s.member else ''}")
    else:
        build_real(a.raw, a.out, a.volumes)