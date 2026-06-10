"""ED28 — the distributed FIFO queue: delivery semantics by consistency model (DS0 increment 21).

The smoke instance of the DS0-increment-21 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — one item, dequeued on both sides of a partition, is
delivered twice under `eventual` (duplicate), once under `quorum` (exactly-once on the majority),
and zero times under `linearizable` (CP unavailable) (Panel A); and on the connected path a queue is
FIFO and exactly-once (Panel B) — with Tier-B reproducing every step. The committed figure is local.
"""

from __future__ import annotations

from verisim.experiments.ed28 import ED28Config, ED28Result, run_ed28, write_csv


def test_panel_a_delivery_count_steps_down_with_the_model():
    result = run_ed28(ED28Config())
    assert result.eventual_deliveries == 2  # at-least-once: the item is delivered on both sides
    assert result.quorum_deliveries == 1  # exactly-once on the majority side
    assert result.linearizable_deliveries == 0  # CP: both sides unavailable, never duplicated


def test_panel_b_fifo_and_exactly_once_on_the_connected_path():
    result = run_ed28(ED28Config())
    assert result.fifo_preserved is True  # dequeue order equals enqueue order
    assert result.exactly_once_connected is True  # each item once, then empty


def test_tier_b_reproduces_every_transition():
    result = run_ed28(ED28Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed28(ED28Config())
    out = write_csv(result, tmp_path / "ed28.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("delivery,") for line in lines)
    assert any(line.startswith("fifo,") for line in lines)


def test_config_round_trips():
    cfg = ED28Config.from_dict({"queue": "jobs", "fifo_items": ["a", "b"]})
    assert cfg.queue == "jobs" and cfg.fifo_items == ("a", "b")
    assert isinstance(run_ed28(cfg), ED28Result)
