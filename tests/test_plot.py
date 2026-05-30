"""The E1 plotting script renders a figure + CSV from run-records (SPEC-2 §16)."""

from __future__ import annotations

import json
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


def _comparison_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    for label in ("fixed", "uncertainty", "drift"):
        for seed in (1, 2, 3):
            records.append(
                RunRecord(
                    config={"policy": label, "rho": 0.5},
                    seed=seed,
                    epsilon=0.0,
                    divergences=[0.0, 0.0, 0.5],
                    consultation_schedule=[True, False, False],
                )
            )
    return records


def test_plot_comparison_writes_figure_and_csv(tmp_path):
    recs = tmp_path / "recs.jsonl"
    write_records(_comparison_records(), recs)
    out = tmp_path / "cmp.png"
    csv = tmp_path / "cmp.csv"
    result = subprocess.run(
        [
            sys.executable, "figures/plot_comparison.py",
            "--records", str(recs), "--key", "policy",
            "--out", str(out), "--csv", str(csv), "--resamples", "50",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 0
    assert csv.exists()
    assert "label,epsilon,mean_h" in csv.read_text()


def test_plot_calibration_writes_figure_and_csv(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    # A monotone signal->divergence relationship so the reliability curve is non-trivial.
    pairs.write_text(
        "\n".join(
            json.dumps({"signal": s / 10.0, "divergence": s / 20.0}) for s in range(11)
        )
        + "\n"
    )
    out = tmp_path / "cal.png"
    csv = tmp_path / "cal.csv"
    result = subprocess.run(
        [
            sys.executable, "figures/plot_calibration.py",
            "--pairs", str(pairs), "--out", str(out), "--csv", str(csv), "--bins", "5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 0
    assert csv.exists()
    text = csv.read_text()
    assert "lo,hi,mean_signal,mean_divergence,n" in text
    assert "pearson,spearman" in text


def _e4_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    for size, acc in (("tiny", 0.1), ("small", 0.3), ("medium", 0.6)):
        for difficulty in ("low", "high"):
            for seed in (1, 2, 3):
                records.append(
                    RunRecord(
                        config={"size": size, "difficulty": difficulty, "step_accuracy": acc},
                        seed=seed,
                        epsilon=0.0,
                        divergences=[0.0],
                    )
                )
    return records


def test_plot_e4_writes_figure_and_csv(tmp_path):
    recs = tmp_path / "recs.jsonl"
    write_records(_e4_records(), recs)
    out = tmp_path / "e4.png"
    csv = tmp_path / "e4.csv"
    result = subprocess.run(
        [
            sys.executable, "figures/plot_e4.py",
            "--records", str(recs), "--out", str(out), "--csv", str(csv), "--resamples", "50",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 0
    assert "size,difficulty,mean_accuracy" in csv.read_text()
