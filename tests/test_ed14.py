"""ED14 — the quorum consensus availability frontier + split-brain prevention (SPEC-7 §3.4, DS0 7).

The smoke instance of the DS0-increment-7 apparatus (dependency-free, GPU-free): a tiny sweep that
checks the two findings have the right shape — quorum's write availability steps at the majority
(Panel A), and only quorum is both available (on the majority side) and split-brain-free (Panel B),
with Tier-B reproducing the decision bit-for-bit. The committed figure comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed14 import ED14Config, ED14Result, run_ed14, write_csv


def test_panel_a_availability_frontier_steps_at_majority():
    result = run_ed14(ED14Config())
    by_model = {r["model"]: r["commits"] for r in result.availability}
    assert result.majority == 3
    # eventual: always available
    assert all(v == 1 for v in by_model["eventual"].values())
    # quorum: available iff k >= 3 (the majority frontier)
    assert by_model["quorum"] == {1: 0, 2: 0, 3: 1, 4: 1}
    # linearizable: unavailable under any partition (needs all replicas)
    assert all(v == 0 for v in by_model["linearizable"].values())


def test_panel_b_only_quorum_is_available_and_fork_free():
    result = run_ed14(ED14Config())
    by_model = {r["model"]: r for r in result.split_brain}
    assert by_model["eventual"]["fork_rate"] == 1.0       # both sides commit → split-brain
    assert by_model["quorum"]["fork_rate"] == 0.0         # only majority commits → no fork
    assert by_model["linearizable"]["fork_rate"] == 0.0   # neither commits → no fork (unavailable)


def test_tier_b_reproduces_the_quorum_decision():
    result = run_ed14(ED14Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed14(ED14Config())
    out = write_csv(result, tmp_path / "ed14.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("availability,") for line in lines)
    assert any(line.startswith("split_brain,") for line in lines)


def test_config_round_trips():
    cfg = ED14Config.from_dict({"n_nodes": 5, "minority_sizes": [1, 2, 3, 4]})
    assert cfg.n_nodes == 5
    assert cfg.minority_sizes == (1, 2, 3, 4)
    assert isinstance(run_ed14(cfg), ED14Result)
