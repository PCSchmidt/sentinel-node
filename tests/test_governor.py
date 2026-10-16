from sentinel_node.governor import (
    BandwidthGovernor,
    Level,
    Load,
    MessageSizes,
    choose_level,
    required_bps,
)

SIZES = MessageSizes(track_bytes=350, thumbnail_bytes=4000)
LOAD = Load(n_tracks=5, update_hz=1.0, reduced_hz=0.2)


def test_required_bps_ordering():
    costs = [required_bps(l, LOAD, SIZES) for l in (Level.OFF, Level.REDUCED, Level.TRACKS, Level.FULL)]
    assert costs[0] == 0.0
    assert costs == sorted(costs) and len(set(costs)) == 4


def test_required_bps_values():
    # 5 tracks * 1 Hz * 350 B * 8 = 14000 bps.
    assert required_bps(Level.TRACKS, LOAD, SIZES) == 14_000.0
    # Reduced: 0.2 Hz -> 2800 bps.
    assert required_bps(Level.REDUCED, LOAD, SIZES) == 2_800.0


def test_no_tracks_costs_nothing():
    assert required_bps(Level.FULL, Load(0, 1.0), SIZES) == 0.0


def test_choose_level_degrades_with_capacity():
    levels = [choose_level(c, LOAD, SIZES) for c in (1_000_000, 40_000, 9_600, 3_000, 500)]
    assert levels == [Level.FULL, Level.TRACKS, Level.REDUCED, Level.OFF, Level.OFF]
    assert levels == sorted(levels, reverse=True)  # never gets richer as capacity falls


def test_known_thresholds():
    # TRACKS needs 14 kbps / 0.7 = 20 kbps of capacity.
    assert choose_level(19_000, LOAD, SIZES) == Level.REDUCED
    assert choose_level(21_000, LOAD, SIZES) == Level.TRACKS
    # REDUCED needs 2.8 kbps / 0.7 = 4 kbps.
    assert choose_level(3_900, LOAD, SIZES) == Level.OFF
    assert choose_level(4_100, LOAD, SIZES) == Level.REDUCED


def test_governor_degrades_immediately():
    g = BandwidthGovernor(SIZES)
    assert g.update(1_000_000, LOAD) == Level.FULL
    assert g.update(21_000, LOAD) == Level.TRACKS
    assert g.update(500, LOAD) == Level.OFF


def test_governor_hysteresis_prevents_flapping():
    g = BandwidthGovernor(SIZES)
    g.update(19_000, LOAD)
    assert g.level == Level.REDUCED
    # 21 kbps would allow TRACKS, but not with the upgrade margin.
    assert g.update(21_000, LOAD) == Level.REDUCED
    # Plenty of slack: upgrade.
    assert g.update(40_000, LOAD) == Level.TRACKS


def test_governor_starts_off_and_climbs():
    g = BandwidthGovernor(SIZES)
    assert g.level == Level.OFF
    assert g.update(100_000, LOAD) >= Level.TRACKS
