"""Digital twin: replays real I/O against a tier placement, one hour at a time.

Every hour:
  1. the strategy picks a tier for every extent, using ONLY past hours
  2. the twin applies the moves (move cost, early-deletion fees, GB moved)
  3. the twin serves that hour's I/Os from the chosen tiers and measures
     storage cost, retrieval cost, latency (with queueing), SLA and compliance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ayojna.contracts import (
    EXTENT_HOURLY_COLUMNS,
    EXTENT_MB,
    TIER_ORDER,
    SimMetrics,
    validate_frame,
)
from ayojna.settings import CONFIG_DIR, load_config

EXTENT_GB = EXTENT_MB / 1024
HOT, WARM, COLD, ARCHIVE = range(4)


def load_twin_config() -> dict:
    with open(Path(CONFIG_DIR) / "twin.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class TwinView:
    """What a strategy is allowed to see at the start of hour `hour`: the past only."""

    hour: int
    ios_past: np.ndarray  # shape (hour, n_extents)
    placement: np.ndarray  # current tier index per extent
    capacity_extents: np.ndarray  # max extents per tier
    volumes: np.ndarray
    extent_ids: np.ndarray


@dataclass
class Twin:
    volumes: np.ndarray
    extent_ids: np.ndarray
    ios: np.ndarray  # shape (hours, n_extents)
    read_bytes: np.ndarray  # shape (hours, n_extents)
    sla_target_ms: np.ndarray  # per extent, from its volume's SLA class
    legal_hold: np.ndarray  # per extent, bool
    price: np.ndarray  # per tier, $ per GB-month
    retrieval: np.ndarray  # per tier, $ per GB read
    min_hours: np.ndarray  # per tier, minimum storage time in hours
    base_latency: np.ndarray  # per tier, ms
    ios_capacity: np.ndarray  # per tier, I/Os per hour
    capacity_extents: np.ndarray
    move_cost_per_gb: float
    hours_per_month: float
    max_util: float
    initial_tier: int
    extra: dict = field(default_factory=dict)

    @property
    def n_hours(self) -> int:
        return self.ios.shape[0]

    @property
    def n_extents(self) -> int:
        return self.ios.shape[1]

    @classmethod
    def from_extent_hourly(cls, eh: pd.DataFrame) -> "Twin":
        eh = validate_frame(eh, EXTENT_HOURLY_COLUMNS, "extent_hourly")
        if eh.empty:
            raise ValueError("extent_hourly is empty")
        cfg, tw = load_config(), load_twin_config()
        # Every extent of every volume, up to the highest one ever touched:
        # data nobody read this week still sits on a tier and still costs money.
        max_ext = eh.groupby("volume")["extent_id"].max()
        keys = pd.DataFrame(
            [(v, e) for v, m in max_ext.items() for e in range(int(m) + 1)],
            columns=["volume", "extent_id"],
        )
        keys["col"] = np.arange(len(keys))
        eh = eh.merge(keys, on=["volume", "extent_id"])
        n_hours, n_ext = int(eh["hour"].max()) + 1, len(keys)
        ios = np.zeros((n_hours, n_ext))
        rbytes = np.zeros((n_hours, n_ext))
        h, c = eh["hour"].to_numpy(), eh["col"].to_numpy()
        np.add.at(ios, (h, c), (eh["reads"] + eh["writes"]).to_numpy())
        np.add.at(rbytes, (h, c), eh["read_bytes"].to_numpy())

        tags = [cfg.tags_for(v) for v in keys["volume"]]
        tiers = [cfg.tiers.tiers[t] for t in TIER_ORDER]
        shares = np.array([t.capacity_share for t in tiers])
        return cls(
            volumes=keys["volume"].to_numpy(),
            extent_ids=keys["extent_id"].to_numpy(),
            ios=ios,
            read_bytes=rbytes,
            sla_target_ms=np.array([cfg.sla[t.sla_class].p95_latency_ms for t in tags]),
            legal_hold=np.array([t.legal_hold for t in tags]),
            price=np.array([t.price_gb_month for t in tiers]),
            retrieval=np.array([t.retrieval_per_gb for t in tiers]),
            min_hours=np.array([t.min_storage_days * 24 for t in tiers]),
            base_latency=np.array([t.base_latency_ms for t in tiers]),
            ios_capacity=np.array(
                [tw["tier_ios_per_hour"][t.value] for t in TIER_ORDER], dtype=float
            ),
            capacity_extents=np.maximum(1, np.floor(shares * n_ext)).astype(int),
            move_cost_per_gb=cfg.tiers.move_cost_per_gb,
            hours_per_month=tw["hours_per_month"],
            max_util=tw["max_utilization"],
            initial_tier=[t.value for t in TIER_ORDER].index(tw["initial_tier"]),
        )

    def run(self, strategy) -> pd.DataFrame:
        """Replay the whole trace under one strategy. Returns one row per hour."""
        placement = np.full(self.n_extents, self.initial_tier)
        since = np.zeros(self.n_extents)
        start_tier = placement.copy()
        rows = []
        for h in range(self.n_hours):
            view = TwinView(
                h,
                self.ios[:h],
                placement.copy(),
                self.capacity_extents,
                self.volumes,
                self.extent_ids,
            )
            target = np.asarray(strategy.decide(view), dtype=int)
            if target.shape != placement.shape or target.min() < 0 or target.max() > ARCHIVE:
                raise ValueError(f"{strategy.name}: invalid placement at hour {h}")

            moved = target != placement
            gb_moved = moved.sum() * EXTENT_GB
            held = h - since
            early = moved & (held < self.min_hours[placement])
            early_fee = np.sum(
                EXTENT_GB
                * self.price[placement[early]]
                * (self.min_hours[placement[early]] - held[early])
                / self.hours_per_month
            )
            placement = target
            since[moved] = h

            storage = np.sum(EXTENT_GB * self.price[placement]) / self.hours_per_month
            retrieval = np.sum(self.read_bytes[h] / 1e9 * self.retrieval[placement])
            move_cost = gb_moved * self.move_cost_per_gb

            ios_h = self.ios[h]
            tier_load = np.bincount(placement, weights=ios_h, minlength=4)
            util = np.minimum(self.max_util, tier_load / self.ios_capacity)
            latency = (self.base_latency / (1 - util))[placement]
            total = ios_h.sum()
            if total > 0:
                sla = 100 * ios_h[latency <= self.sla_target_ms].sum() / total
                order = np.argsort(latency)
                cum = np.cumsum(ios_h[order]) / total
                p95 = float(latency[order][np.searchsorted(cum, 0.95)])
            else:
                sla, p95 = 100.0, 0.0
            violations = np.sum(self.legal_hold & (placement != start_tier))
            compliance = 100 * (1 - violations / self.n_extents)

            m = SimMetrics(
                strategy=strategy.name,
                hour=h,
                cost=storage + retrieval + move_cost + early_fee,
                p95_latency_ms=p95,
                sla_met_pct=sla,
                gb_moved=gb_moved,
                compliance_pct=compliance,
            )
            rows.append(
                {
                    **m.model_dump(),
                    "storage_cost": storage,
                    "retrieval_cost": retrieval,
                    "move_cost": move_cost + early_fee,
                    "hot_share": float(np.mean(placement == HOT)),
                    # I/O accounting for the KPI report (hit ratio, I/O-weighted SLA)
                    "ios": float(total),
                    "hot_ios": float(ios_h[placement == HOT].sum()),
                    "sla_ios": float(ios_h[latency <= self.sla_target_ms].sum()),
                    "hot_over_capacity": bool(
                        np.sum(placement == HOT) > self.capacity_extents[HOT]
                    ),
                }
            )
        return pd.DataFrame(rows)
