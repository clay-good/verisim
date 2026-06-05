"""SPEC-10 cross-proposer synthesis tests (the floor-is-proposer-dependent capstone, H26).

Pure figures-from-records: the synthesis re-reads two committed capacity-sweep CSVs and overlays
their free-running horizon on the shared ``params`` axis. No torch, no training -- so these tests
are fast and ungated. They write tiny synthetic CSVs and assert the overlay is well-formed and that
the flat arm's lift vs the graph arm's pinned floor is read correctly.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.horizon_synthesis import (
    CSV_HEADER,
    cross_proposer_synthesis,
    write_csv,
)

_FLAT_CSV = """scale,params,metric,mean,ci_lo,ci_hi,n
xs,1024,h_free_id,1.75,0.75,2.50,3
xs,1024,h_free_ood,1.92,1.0,3.0,3
xs,1024,one_step_acc_id,0.47,0.4,0.5,3
m,32768,h_free_id,15.83,14.25,16.75,3
m,32768,h_free_ood,17.42,16.0,18.0,3
m,32768,one_step_acc_id,0.82,0.8,0.85,3
"""

_GRAPH_CSV = """scale,params,metric,mean,ci_lo,ci_hi,n
xs,1024,h_free_id,0.0,0.0,0.0,3
xs,1024,h_free_ood,0.0,0.0,0.0,3
xs,1024,one_step_acc_id,0.64,0.6,0.66,3
m,32768,h_free_id,0.0,0.0,0.0,3
m,32768,h_free_ood,0.0,0.0,0.0,3
m,32768,one_step_acc_id,0.67,0.65,0.69,3
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_cross_proposer_synthesis_overlays_both(tmp_path: Path) -> None:
    flat = _write(tmp_path, "flat.csv", _FLAT_CSV)
    graph = _write(tmp_path, "graph.csv", _GRAPH_CSV)
    points = cross_proposer_synthesis({"flat (HS1)": flat, "graph (HS3)": graph})

    # both proposers, both capacities, sorted by params within proposer
    assert {p.proposer for p in points} == {"flat (HS1)", "graph (HS3)"}
    flat_pts = [p for p in points if p.proposer == "flat (HS1)"]
    graph_pts = [p for p in points if p.proposer == "graph (HS3)"]
    assert [p.params for p in flat_pts] == [1024, 32768]

    # the flat arm lifts with capacity; the graph arm stays pinned at the floor
    assert flat_pts[0].h_free_id == 1.75 and flat_pts[1].h_free_id == 15.83
    assert flat_pts[1].h_free_id > flat_pts[0].h_free_id
    assert all(p.h_free_id == 0.0 for p in graph_pts)
    # but the graph arm's per-step accuracy is healthy (the proxy/truth divergence)
    assert graph_pts[1].p_id > 0.6


def test_write_csv_roundtrips(tmp_path: Path) -> None:
    flat = _write(tmp_path, "flat.csv", _FLAT_CSV)
    graph = _write(tmp_path, "graph.csv", _GRAPH_CSV)
    points = cross_proposer_synthesis({"flat": flat, "graph": graph})
    out = write_csv(points, tmp_path / "synth.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0] == CSV_HEADER
    assert len(lines) == 1 + len(points)
