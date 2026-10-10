"""Baseline tiering strategies. Each sees only the past (a TwinView) and returns
a tier index per extent: 0 hot, 1 warm, 2 cold, 3 archive.

Ayojna's own planner will plug in later through the same `decide(view)` method.
"""

from __future__ import annotations

import numpy as np

from ayojna.twin.sim import COLD, HOT, WARM, TwinView


def last_access_hour(view: TwinView) -> np.ndarray:
    """Hour of each extent's most recent I/O, or -1 if never seen yet."""
    if view.hour == 0:
        return np.full(view.placement.shape, -1)
    touched = view.ios_past > 0
    any_touch = touched.any(axis=0)
    last = view.hour - 1 - np.argmax(touched[::-1], axis=0)
    return np.where(any_touch, last, -1)


class AllHot:
    """Keep everything on the fastest tier: the cost ceiling."""

    name = "all_hot"

    def decide(self, view: TwinView) -> np.ndarray:
        return np.full(view.placement.shape, HOT)


class AgeRule:
    """Today's common practice: demote data after it has been idle for a while."""

    name = "age_rule"

    def __init__(self, warm_after_idle_hours: int = 24, cold_after_idle_hours: int = 72):
        self.warm_after = warm_after_idle_hours
        self.cold_after = cold_after_idle_hours

    def decide(self, view: TwinView) -> np.ndarray:
        if view.hour == 0:
            return view.placement
        last = last_access_hour(view)
        idle = np.where(last >= 0, view.hour - last, view.hour + 1)
        return np.select(
            [idle < self.warm_after, idle < self.cold_after], [HOT, WARM], default=COLD
        )


class AccessTimer:
    """Cloud-style auto-tiering: demote after a fixed idle time, promote on any access.

    Only uses hot and warm (no retrieval or early-deletion fees), like most
    cloud auto-tiering classes. The idle timer is scaled to a week-long trace.
    """

    name = "access_timer"

    def __init__(self, idle_hours: int = 24):
        self.idle_hours = idle_hours

    def decide(self, view: TwinView) -> np.ndarray:
        if view.hour == 0:
            return view.placement
        last = last_access_hour(view)
        idle = np.where(last >= 0, view.hour - last, view.hour + 1)
        return np.where(idle < self.idle_hours, HOT, WARM)


class LruCapacity:
    """Most recently used extents fill the hot tier, then warm; the rest go cold."""

    name = "lru"

    def decide(self, view: TwinView) -> np.ndarray:
        if view.hour == 0:
            return view.placement
        last = last_access_hour(view)
        order = np.argsort(-last, kind="stable")  # most recent first
        tiers = np.full(view.placement.shape, COLD)
        n_hot, n_warm = view.capacity_extents[HOT], view.capacity_extents[WARM]
        tiers[order[:n_hot]] = HOT
        tiers[order[n_hot : n_hot + n_warm]] = WARM
        tiers[last < 0] = np.where(view.placement[last < 0] == HOT, COLD, view.placement[last < 0])
        return tiers


class LfuCapacity:
    """Most FREQUENTLY used extents (I/Os in a sliding window) fill hot, then warm; rest cold.

    The window keeps it from clinging to data that was busy long ago (LFU with ageing).
    Ties are broken by recency, like LRU.
    """

    name = "lfu"

    def __init__(self, window_hours: int = 72):
        self.window = window_hours

    def decide(self, view: TwinView) -> np.ndarray:
        if view.hour == 0:
            return view.placement
        freq = view.ios_past[-self.window :].sum(axis=0)
        last = last_access_hour(view)
        order = np.lexsort((-last, -freq))  # most I/Os first, then most recent
        tiers = np.full(view.placement.shape, COLD)
        n_hot, n_warm = view.capacity_extents[HOT], view.capacity_extents[WARM]
        tiers[order[:n_hot]] = HOT
        tiers[order[n_hot : n_hot + n_warm]] = WARM
        tiers[last < 0] = np.where(view.placement[last < 0] == HOT, COLD, view.placement[last < 0])
        return tiers


class PolicyAware:
    """Any baseline made compliant: each choice snaps to the nearest tier the policy allows.

    Rules that ignore policy look cheaper by breaking it (moving legal-hold data, archiving
    PII). Wrapping them in the same policy guard Ayojna uses gives the fair comparison.
    """

    def __init__(self, inner):
        self.inner, self.name = inner, f"{inner.name}+policy"

    def decide(self, view: TwinView) -> np.ndarray:
        from ayojna.policy.guard import allowed_tiers  # here: the twin does not need policy

        want = np.asarray(self.inner.decide(view))
        ok, _ = allowed_tiers(view.volumes, view.placement)
        tiers = np.arange(ok.shape[1])
        # distance to the wanted tier; forbidden tiers never win; ties go to the faster tier
        dist = np.abs(tiers[None, :] - want[:, None]) + np.where(ok, 0, 99) + tiers[None, :] * 1e-3
        return dist.argmin(axis=1)
