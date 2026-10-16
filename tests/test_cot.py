import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pytest

from sentinel_node.cot import GeoTrack, TYPE_UNKNOWN_GROUND, to_cot_xml

NOW = datetime(2026, 10, 20, 15, 30, 5, 123000, tzinfo=timezone.utc)


def parse(track, **kw):
    return ET.fromstring(to_cot_xml(track, NOW, **kw))


def test_event_structure_and_attributes():
    ev = parse(GeoTrack("sn-1", 38.9, -77.0, hae_m=12.5, course_deg=90.0, speed_mps=14.2))
    assert ev.tag == "event"
    assert ev.get("version") == "2.0"
    assert ev.get("uid") == "sn-1"
    assert ev.get("type") == "a-u-A"
    assert ev.get("how") == "m-g"
    pt = ev.find("point")
    assert float(pt.get("lat")) == pytest.approx(38.9)
    assert float(pt.get("lon")) == pytest.approx(-77.0)
    assert float(pt.get("hae")) == pytest.approx(12.5)
    trk = ev.find("detail/track")
    assert float(trk.get("course")) == pytest.approx(90.0)
    assert float(trk.get("speed")) == pytest.approx(14.2)


def test_times_are_utc_millisecond_iso_and_stale_is_later():
    ev = parse(GeoTrack("a", 0.0, 0.0), stale_s=45.0)
    assert ev.get("time") == "2026-10-20T15:30:05.123Z"
    assert ev.get("start") == ev.get("time")
    assert ev.get("stale") == "2026-10-20T15:30:50.123Z"


def test_non_utc_input_is_converted_to_utc():
    est = timezone(timedelta(hours=-4))
    ev = ET.fromstring(to_cot_xml(GeoTrack("a", 0, 0), NOW.astimezone(est)))
    assert ev.get("time").endswith("Z") and ev.get("time").startswith("2026-10-20T15:30:05")


def test_course_is_wrapped_and_type_overridable():
    ev = parse(GeoTrack("g", 1.0, 1.0, course_deg=-90.0, cot_type=TYPE_UNKNOWN_GROUND))
    assert float(ev.find("detail/track").get("course")) == pytest.approx(270.0)
    assert ev.get("type") == "a-u-G"


@pytest.mark.parametrize("lat,lon", [(91.0, 0.0), (-90.1, 0.0), (0.0, 181.0), (0.0, -180.5)])
def test_out_of_range_coordinates_rejected(lat, lon):
    with pytest.raises(ValueError):
        to_cot_xml(GeoTrack("x", lat, lon), NOW)


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        to_cot_xml(GeoTrack("x", 0.0, 0.0), datetime(2026, 1, 1))


def test_message_is_small():
    # The whole point is to send tracks instead of video.
    assert len(to_cot_xml(GeoTrack("sn-12", 38.9, -77.0, hae_m=10, course_deg=45, speed_mps=12), NOW)) < 600
