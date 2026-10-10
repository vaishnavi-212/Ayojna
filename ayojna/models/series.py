"""Per-volume hourly series, shared by the forecast and anomaly models.

One row per (volume, hour), every hour present (zeros where the volume was idle).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ayojna.contracts import EXTENT_MB

GB_PER_EXTENT = EXTENT_MB / 1024


def volume_hourly(eh: pd.DataFrame) -> pd.DataFrame:
    df = eh.assign(
        io=eh["reads"] + eh["writes"],
        bytes=eh["read_bytes"] + eh["write_bytes"],
        rand_io=eh["rand_ratio"] * (eh["reads"] + eh["writes"]),
    )
    agg = df.groupby(["volume", "hour"]).agg(
        io=("io", "sum"),
        reads=("reads", "sum"),
        bytes=("bytes", "sum"),
        rand_io=("rand_io", "sum"),
        extents=("extent_id", "nunique"),
    )
    hours = np.arange(int(eh["hour"].max()) + 1)
    full = pd.MultiIndex.from_product([sorted(eh["volume"].unique()), hours], names=agg.index.names)
    out = agg.reindex(full, fill_value=0).reset_index()
    safe = out["io"].replace(0, np.nan)
    out["read_ratio"] = (out["reads"] / safe).fillna(0.0)
    out["avg_io_size"] = (out["bytes"] / safe).fillna(0.0)
    out["rand_ratio"] = (out["rand_io"] / safe).fillna(0.0)
    out["ws_gb"] = out["extents"] * GB_PER_EXTENT  # working set: GB of extents touched
    return out