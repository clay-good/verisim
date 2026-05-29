"""The E1 plotting script renders a figure + CSV from run-records (SPEC-2 §16)."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")

from verisim.metrics import RunRecord, write_records


def _synthetic_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    for rho in (0.0, 0.5, 1.0):
        for seed in (1, 2, 3):
            faithful = int(rho * 6)
            records.append(
                RunRecord(
                    config={"difficulty": "low", "rho": rho},
                    seed=seed,
                    epsilon=0.0,
                    divergences=[0.0] * faithful + [0.5],
                )
            )
    return records


def test_plot_e1_writes_figure_and_csv(tmp_path):
    recs = tmp_path / "recs.jsonl"
    write_records(_synthetic_records(), recs)
    out = tmp_path / "curve.png"
    csv = tmp_path / "curve.csv"
    result = subprocess.run(
        [
            sys.executable, "figures/plot_e1.py",
            "--records", str(recs), "--out", str(out), "--csv", str(csv), "--resamples", "50",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 0
    assert csv.exists()
    assert "difficulty,epsilon,rho" in csv.read_text()
