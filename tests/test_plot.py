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


def _auto_records() -> list[RunRecord]:
    # A small ratchet: score rises, then a rejected trial; best_score is monotone.
    rows = [(0, 0.1, 0.1, True), (1, 0.2, 0.2, True), (2, 0.15, 0.2, False)]
    return [
        RunRecord(
            config={
                "experiment": "auto-search", "trial": t, "score": s, "best_score": b, "kept": k,
                "n_layer": 1, "n_embd": 32, "train_iters": 200, "train_steps_per_traj": 20,
                "lr": 0.001,
            },
            seed=0,
            epsilon=0.0,
            divergences=[],
        )
        for t, s, b, k in rows
    ]


def test_plot_auto_writes_figure_and_csv(tmp_path):
    recs = tmp_path / "log.jsonl"
    write_records(_auto_records(), recs)
    out = tmp_path / "auto.png"
    csv = tmp_path / "auto.csv"
    result = subprocess.run(
        [
            sys.executable, "figures/plot_auto.py",
            "--records", str(recs), "--out", str(out), "--csv", str(csv),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 0
    assert "trial,score,best_score,kept" in csv.read_text()


def _k0_records() -> list[RunRecord]:
    control = RunRecord(
        config={
            "experiment": "k0", "part": "control", "world": "trivial",
            "clean_faithfulness_exact": 0.96, "clean_faithfulness_graded": 0.99,
            "gate": 0.95, "gate_passed": True, "n_train_transitions": 384,
        },
        seed=0, epsilon=0.0, divergences=[],
    )
    diagnostics = RunRecord(
        config={
            "experiment": "k0", "part": "diagnostics",
            "baseline_train_accuracy": 0.9, "baseline_val_accuracy": 0.12,
            "baseline_mean_bits_to_correct": 18.0,
            "per_command": {"write": [3, 10], "mkdir": [6, 10], "cat": [0, 5]},
            "per_edit_pr": {"create": [5, 8, 9]},
            "accuracy_by_position": [[2, 3], [1, 3]],
        },
        seed=0, epsilon=0.0, divergences=[],
    )
    return [control, diagnostics]


def test_plot_k0_writes_figure_and_csv(tmp_path):
    recs = tmp_path / "k0.jsonl"
    write_records(_k0_records(), recs)
    out = tmp_path / "k0.png"
    csv = tmp_path / "k0.csv"
    result = subprocess.run(
        [
            sys.executable, "figures/plot_k0.py",
            "--records", str(recs), "--out", str(out), "--csv", str(csv),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 0
    text = csv.read_text()
    assert "trivial_control_exact" in text
    assert "baseline_val_accuracy" in text
