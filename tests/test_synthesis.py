"""Cross-world synthesis tests (the paper's floor+cliff-across-worlds figure).

Torch-free: the synthesis only reads committed curve CSVs and normalizes. Tests the aggregation on
small temp CSVs (difficulty averaging, per-world normalization by the ρ=1 ceiling, the ED1 panel
schema) and confirms the four committed world CSVs load and overlay.
"""

from __future__ import annotations

import csv

from verisim.experiments.synthesis import (
    DEFAULT_WORLDS,
    cross_world_curve,
    world_curve,
)


def _write_csv(path, rows):
    with open(path, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["difficulty", "epsilon", "rho", "mean_h", "ci_low", "ci_high", "n"])
        for r in rows:
            w.writerow(r)


def test_world_curve_averages_over_difficulty_at_epsilon(tmp_path):
    csv_path = tmp_path / "w.csv"
    _write_csv(csv_path, [
        ("low", 0.0, 0.0, 2.0, 0, 0, 5), ("high", 0.0, 0.0, 4.0, 0, 0, 5),  # avg 3.0 at rho 0
        ("low", 0.0, 1.0, 10.0, 0, 0, 5), ("high", 0.0, 1.0, 14.0, 0, 0, 5),  # avg 12.0 at rho 1
        ("low", 0.1, 0.0, 99.0, 0, 0, 5),  # different epsilon -> ignored
    ])
    curve = world_curve(csv_path, epsilon=0.0)
    assert curve == [(0.0, 3.0), (1.0, 12.0)]


def test_world_curve_reads_ed1_panel_schema(tmp_path):
    # The distributed world (ED1) emits the panel schema; world_curve reads its `curve` rows as
    # (x=ρ, h_eps), ignoring the h17 panel rows and the epsilon argument (the curve is the faithful
    # bit-exact horizon, not selected by ε).
    csv_path = tmp_path / "ed1.csv"
    with open(csv_path, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["panel", "key", "x", "tier", "mode", "h_eps", "dollars",
                    "dollars_per_step", "ci_lo", "ci_hi"])
        w.writerow(["curve", "rho", 0.0, "bit_exact", "gross", 0.5, "", "", 0.0, 1.0])
        w.writerow(["curve", "rho", 1.0, "bit_exact", "gross", 10.0, "", "", 10.0, 10.0])
        w.writerow(["h17", "gross/meta", "", "metamorphic", "gross", 16.0, 152.0, 9.3, "", ""])
    curve = world_curve(csv_path, epsilon=0.0)
    assert curve == [(0.0, 0.5), (1.0, 10.0)]  # only the two curve rows, h17 row ignored


def test_cross_world_normalizes_by_each_worlds_ceiling(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, [("low", 0.0, 0.0, 1.0, 0, 0, 5), ("low", 0.0, 1.0, 10.0, 0, 0, 5)])
    _write_csv(b, [("low", 0.0, 0.0, 4.0, 0, 0, 5), ("low", 0.0, 1.0, 8.0, 0, 0, 5)])
    points = cross_world_curve({"A": a, "B": b}, epsilon=0.0)
    frac = {(p.world, p.rho): p.frac_h for p in points}
    # each world normalized by its OWN ceiling (A:10, B:8) -> ρ=1 is 1.0 for both, floors differ
    assert frac[("A", 0.0)] == 0.1 and frac[("A", 1.0)] == 1.0
    assert frac[("B", 0.0)] == 0.5 and frac[("B", 1.0)] == 1.0


def test_committed_world_csvs_load_and_overlay():
    points = cross_world_curve(epsilon=0.0)
    worlds = {p.world for p in points}
    assert worlds == set(DEFAULT_WORLDS)
    # every world reaches its ceiling at ρ=1 and sits below it across the interior (the floor+cliff)
    for world in worlds:
        cells = {p.rho: p.frac_h for p in points if p.world == world}
        assert abs(cells[1.0] - 1.0) < 1e-9
        assert cells[0.0] < 0.5  # a real floor at ρ=0 (drift), well below the ceiling
