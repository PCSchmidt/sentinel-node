"""Bandwidth governor: degrade what the node publishes as the link degrades.

Levels, richest first:

* ``FULL``     tracks plus a small thumbnail per confirmed track
* ``TRACKS``   tracks at the normal update rate
* ``REDUCED``  tracks decimated to fit the link
* ``OFF``      nothing (link cannot carry even decimated tracks)

The governor picks the richest level whose steady-state cost fits within a
fraction of the measured link capacity, with hysteresis so the level does not
flap when capacity hovers near a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Level(IntEnum):
    OFF = 0
    REDUCED = 1
    TRACKS = 2
    FULL = 3


@dataclass(frozen=True)
class MessageSizes:
    """Approximate on-the-wire bytes per message."""

    track_bytes: int = 350
    thumbnail_bytes: int = 4000


@dataclass(frozen=True)
class Load:
    n_tracks: int
    update_hz: float
    reduced_hz: float = 0.2


def required_bps(level: Level, load: Load, sizes: MessageSizes) -> float:
    """Steady-state bits per second needed to publish at ``level``."""
    if level == Level.OFF or load.n_tracks == 0:
        return 0.0
    track_rate = {
        Level.REDUCED: load.reduced_hz,
        Level.TRACKS: load.update_hz,
        Level.FULL: load.update_hz,
    }[level]
    per_s = load.n_tracks * track_rate * sizes.track_bytes
    if level == Level.FULL:
        # One thumbnail per track every ten seconds.
        per_s += load.n_tracks * 0.1 * sizes.thumbnail_bytes
    return per_s * 8.0


def choose_level(capacity_bps: float, load: Load, sizes: MessageSizes, headroom: float = 0.7) -> Level:
    """Richest level whose cost fits in ``headroom * capacity_bps``."""
    budget = headroom * capacity_bps
    for level in (Level.FULL, Level.TRACKS, Level.REDUCED):
        if required_bps(level, load, sizes) <= budget:
            return level
    return Level.OFF


class BandwidthGovernor:
    """Stateful wrapper that adds hysteresis to :func:`choose_level`.

    Degrading is immediate (a link that cannot carry the current level must
    shed load now). Upgrading requires the richer level to fit with an extra
    ``upgrade_margin`` of slack, which stops flapping near a threshold.
    """

    def __init__(self, sizes: MessageSizes | None = None, headroom: float = 0.7, upgrade_margin: float = 0.25):
        self.sizes = sizes or MessageSizes()
        self.headroom = headroom
        self.upgrade_margin = upgrade_margin
        self.level = Level.OFF

    def update(self, capacity_bps: float, load: Load) -> Level:
        target = choose_level(capacity_bps, load, self.sizes, self.headroom)
        if target < self.level:
            self.level = target
        elif target > self.level:
            # Only step up if it still fits with margin.
            conservative = choose_level(
                capacity_bps, load, self.sizes, self.headroom * (1.0 - self.upgrade_margin)
            )
            if conservative > self.level:
                self.level = conservative
        return self.level
