"""Cursor-on-Target (CoT) event encoding for tracks.

Produces a minimal CoT 2.0 ``<event>`` with a ``<point>`` and a ``<track>``
detail, which ATAK/WinTAK-style clients can ingest. Only the fields below are
emitted; richer detail (remarks, sensor links) is not implemented.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Common CoT type strings: a = atom, u = unknown affiliation, then dimension.
TYPE_UNKNOWN_AIR = "a-u-A"
TYPE_UNKNOWN_GROUND = "a-u-G"

# CoT convention for "unknown" circular/linear error and height.
UNKNOWN_ERROR_M = 9999999.0


@dataclass(frozen=True)
class GeoTrack:
    """A track in geodetic coordinates, ready to publish."""

    uid: str
    lat: float
    lon: float
    hae_m: float = 0.0
    course_deg: float = 0.0
    speed_mps: float = 0.0
    cot_type: str = TYPE_UNKNOWN_AIR
    ce_m: float = UNKNOWN_ERROR_M
    le_m: float = UNKNOWN_ERROR_M


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def to_cot_xml(track: GeoTrack, now: datetime, stale_s: float = 30.0) -> bytes:
    """Encode ``track`` as a CoT event valid from ``now`` for ``stale_s`` seconds."""
    if not -90.0 <= track.lat <= 90.0:
        raise ValueError(f"latitude out of range: {track.lat}")
    if not -180.0 <= track.lon <= 180.0:
        raise ValueError(f"longitude out of range: {track.lon}")
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware")

    event = ET.Element(
        "event",
        {
            "version": "2.0",
            "uid": track.uid,
            "type": track.cot_type,
            "how": "m-g",  # machine-generated, GPS-derived position
            "time": _iso(now),
            "start": _iso(now),
            "stale": _iso(now + timedelta(seconds=stale_s)),
        },
    )
    ET.SubElement(
        event,
        "point",
        {
            "lat": f"{track.lat:.7f}",
            "lon": f"{track.lon:.7f}",
            "hae": f"{track.hae_m:.1f}",
            "ce": f"{track.ce_m:.1f}",
            "le": f"{track.le_m:.1f}",
        },
    )
    detail = ET.SubElement(event, "detail")
    ET.SubElement(
        detail,
        "track",
        {"course": f"{track.course_deg % 360.0:.1f}", "speed": f"{track.speed_mps:.1f}"},
    )
    return ET.tostring(event, encoding="utf-8")
