"""ED25 — leader leases: local reads without a quorum + the lease/election tension (DS0 incr 18).

The smoke instance of the DS0-increment-18 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — a live lease lets the leader serve a local linearizable read
with no quorum (even when partitioned into the minority, where its `propose` is `no_quorum`), and an
expired lease rejects (Panel A); a live lease fences a fresh election (`lease_held`) until it
expires, while a voluntary `step_down` releases it for an immediate handoff (Panel B) — with Tier-B
reproducing every transition bit-for-bit. The committed figure comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed25 import ED25Config, ED25Result, run_ed25, write_csv


def test_panel_a_local_reads_without_a_quorum():
    result = run_ed25(ED25Config())
    assert result.valid_lease_read_rate == 1.0  # a live lease serves a local read
    assert result.minority_lread_rate == 1.0  # a minority-stranded leader still reads locally...
    assert result.minority_propose_blocked_rate == 1.0  # ...where its propose is no_quorum
    assert result.expired_lease_read_rate == 1.0  # once expired, the local read is rejected


def test_panel_b_lease_election_safety_tension():
    result = run_ed25(ED25Config())
    assert result.elect_blocked_under_lease is True  # a live lease fences a fresh election
    assert result.elect_after_expiry_ok is True  # ...unblocked once the deadline passes
    assert result.stepdown_releases_lease is True  # step_down releases the lease (fast handoff)


def test_tier_b_reproduces_every_transition():
    result = run_ed25(ED25Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed25(ED25Config())
    out = write_csv(result, tmp_path / "ed25.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("local_read,") for line in lines)
    assert any(line.startswith("safety,") for line in lines)


def test_config_round_trips():
    cfg = ED25Config.from_dict({"cluster_sizes": [3, 5], "lease_dt": 4})
    assert cfg.cluster_sizes == (3, 5) and cfg.lease_dt == 4
    assert isinstance(run_ed25(cfg), ED25Result)
