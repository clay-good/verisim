"""ED4 (consistency-level arm) — weaker consistency opens the H19 gap (SPEC-7 §12; H20/H19).

The smoke instance (dependency-free, GPU-free): a tiny seeded sweep checking the cross-consistency
result has the right qualitative shape — the `subtle` (in-flight) error class opens a
consistency-vs-bit gap under the weak `eventual` model and the gap collapses under the strong
`linearizable` model (where the in-flight medium is structurally absent), while the `gross`
(durable-replica) control shows no gap at either level. The committed figure is from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed4_consistency import (
    ED4ConsistencyConfig,
    ED4ConsistencyResult,
    run_ed4_consistency,
    write_csv,
)


def _tiny() -> ED4ConsistencyConfig:
    return ED4ConsistencyConfig(eval_seeds=(100, 101, 102, 103), n_steps=24)


def test_run_ed4_consistency_is_well_formed():
    result = run_ed4_consistency(_tiny())
    assert {(r["level"], r["mode"]) for r in result.rows} == {
        (lv, m) for lv in ("linearizable", "eventual") for m in ("gross", "subtle")
    }
    assert {v["mode"] for v in result.verdict} == {"gross", "subtle"}


def test_linearizable_has_no_inflight_medium():
    result = run_ed4_consistency(_tiny())
    by = {(r["level"], r["mode"]): r for r in result.rows}
    # synchronous replication enqueues nothing: the consistency-invisible medium is structurally
    # absent under linearizable, present under eventual (async replication).
    assert by[("linearizable", "subtle")]["inflight_rate"] == 0.0
    assert by[("eventual", "subtle")]["inflight_rate"] > 0.0


def test_subtle_gap_is_a_weak_consistency_phenomenon():
    result = run_ed4_consistency(_tiny())
    by = {(r["level"], r["mode"]): r for r in result.rows}
    # eventual + subtle: the consistency-faithful horizon outlasts the bit-faithful one (H19)...
    assert by[("eventual", "subtle")]["cons_h"] > by[("eventual", "subtle")]["bit_h"]
    # ...and the gap collapses under strong consistency (no in-flight medium to hide errors in).
    assert by[("eventual", "subtle")]["gap"] > by[("linearizable", "subtle")]["gap"]
    assert by[("linearizable", "subtle")]["gap"] == 0.0


def test_gross_is_the_control_no_gap_at_either_level():
    result = run_ed4_consistency(_tiny())
    by = {(r["level"], r["mode"]): r for r in result.rows}
    # a durable-replica error is immediately consistency-visible, so bit and consistency coincide
    assert by[("eventual", "gross")]["gap"] == 0.0
    assert by[("linearizable", "gross")]["gap"] == 0.0


def test_write_csv(tmp_path):
    result = run_ed4_consistency(_tiny())
    out = write_csv(result, tmp_path / "ed4c.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("row,") for line in lines)
    assert any(line.startswith("verdict,") for line in lines)


def test_config_round_trips():
    cfg = ED4ConsistencyConfig.from_dict({"noise": 0.3, "levels": ["eventual"]})
    assert cfg.noise == 0.3
    assert cfg.levels == ("eventual",)
    assert isinstance(run_ed4_consistency(_tiny()), ED4ConsistencyResult)
