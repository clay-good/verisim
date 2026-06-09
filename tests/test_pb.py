"""Smoke + structural-invariant tests for the SPEC-18 product experiments (PB-bench/transfer/pack).

PB-bench and PB-pack are CPU-only (controlled-stand-in proposers; trained arms deferred).
PB-transfer needs the SPEC-11 system oracle (real shell) and is ``skipif``-guarded — a skip is never
a result (§2.5). Tests assert structural claims on tiny configs, not magnitudes (macOS-first).
"""

from __future__ import annotations

import pytest

from verisim.experiments.pb_bench import PBBenchConfig, run_pb_bench
from verisim.experiments.pb_pack import PBPackConfig, emit_metadata, run_pb_pack
from verisim.experiments.pb_transfer import (
    PBTransferConfig,
    run_pb_transfer,
    system_oracle_available,
)


def test_pb_bench_discriminative() -> None:
    manifest, rows, stability = run_pb_bench(PBBenchConfig(seeds=16, n_steps=60))
    assert "verisim-bench@" in manifest.version_tag()
    assert len(rows) == len(manifest.proposers) * len(manifest.worlds)
    for s in stability:
        assert s.tau_mean >= 0.8 and s.discriminative  # H65


def test_pb_pack_overfit_detector_and_conformance() -> None:
    result = run_pb_pack(PBPackConfig(public_seeds=12, heldout_seeds=12, n_steps=60))
    # H68: the memorizer's public-minus-held-out gap is far larger than the honest proposer's.
    assert result.memorizer_gap > result.honest_gap + 0.2
    # The conformance suite is green.
    assert result.conformance_pass == result.conformance_total > 0


def test_pb_pack_emits_metadata(tmp_path: object) -> None:
    paths = emit_metadata(PBPackConfig(bench_dir=str(tmp_path)))
    for key in ("croissant", "datasheet", "model_card"):
        assert paths[key]
        from pathlib import Path

        assert Path(paths[key]).exists()


def test_pb_transfer_gap_is_measurable_or_skipped() -> None:
    if not system_oracle_available():
        pytest.skip("system oracle (real shell) unavailable — transfer not counted (§2.5)")
    stats = run_pb_transfer(PBTransferConfig(n_steps=16, n_seeds=4, rhos=(0.0, 0.5)))
    assert stats  # measurable when the shell is present
    # On the validated structure grammar the gap is near zero (lossless transfer, H66) and the
    # correction lifts the absolute real-OS horizon (H67).
    assert abs(stats[0].delta_h) <= 1.0
    assert stats[-1].h_sys >= stats[0].h_sys


def test_pb_transfer_skips_cleanly_when_unavailable() -> None:
    # run_pb_transfer returns an empty list (never a counted result) when the oracle is absent.
    # When present it returns rows; either way the call does not raise.
    stats = run_pb_transfer(PBTransferConfig(n_steps=12, n_seeds=2, rhos=(0.0,)))
    assert isinstance(stats, list)
