"""ED23 — leader election with terms: no split-brain + the fence quorum lacks (DS0 increment 16).

The smoke instance of the DS0-increment-16 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — only a strict-majority side can elect (no split-brain, with
the even-split cluster left leaderless rather than forked, Panel A), and a deposed leader is fenced
after heal while a plain put by the same stale coordinator still commits (term-fencing, the property
a leaderless quorum write lacks, Panel B) — with Tier-B reproducing every transition bit-for-bit.
The committed figure comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed23 import ED23Config, ED23Result, run_ed23, write_csv


def test_panel_a_no_split_brain_only_majority_elects():
    result = run_ed23(ED23Config())
    assert result.minority_elect_blocked_rate == 1.0  # a minority side can never elect
    assert result.majority_elect_rate == 1.0  # a strict-majority side always can
    assert result.even_split_leaderless_rate == 1.0  # a 2|2 leaves NO leader (not two)


def test_panel_b_term_fencing_is_the_property_quorum_lacks():
    result = run_ed23(ED23Config())
    assert result.deposed_propose_fenced is True  # the deposed leader cannot commit after heal
    assert result.unfenced_put_commits is True  # ...but a plain put by the same node still would
    assert result.new_leader_commits is True  # the legitimate new leader commits


def test_tier_b_reproduces_every_transition():
    result = run_ed23(ED23Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed23(ED23Config())
    out = write_csv(result, tmp_path / "ed23.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("split_brain,") for line in lines)
    assert any(line.startswith("fencing,") for line in lines)


def test_config_round_trips():
    cfg = ED23Config.from_dict({"cluster_sizes": [3, 5], "v_old": "b", "v_new": "c"})
    assert cfg.cluster_sizes == (3, 5)
    assert isinstance(run_ed23(cfg), ED23Result)
