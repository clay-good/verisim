"""Tests for the K0 diagnostics battery (SPEC-2.1 §4) on a tiny config."""

from verisim.experiments.diagnose import run_diagnostics
from verisim.experiments.e1 import E1Config

_VALID_COMMANDS = {
    "mkdir", "rmdir", "touch", "cd", "cat", "ls", "rm", "mv", "cp", "write",
    "append", "chmod", "export",
}


def _tiny_config() -> E1Config:
    return E1Config(
        train_seeds=(0, 1),
        train_steps_per_traj=6,
        train_iters=20,
        n_layer=1,
        n_embd=32,
        eval_seeds=(100,),
        eval_steps=6,
        difficulties={"low": "weighted"},
    )


def test_run_diagnostics_structure():
    report = run_diagnostics(_tiny_config())

    assert 0.0 <= report.train_accuracy <= 1.0
    assert 0.0 <= report.val_accuracy <= 1.0
    assert report.mean_bits_to_correct >= 0.0

    assert report.per_command
    for cmd, (correct, total) in report.per_command.items():
        assert cmd in _VALID_COMMANDS
        assert total > 0
        assert 0 <= correct <= total

    for _op, (tp, predicted, true) in report.per_edit_pr.items():
        assert tp <= predicted
        assert tp <= true

    # one position bucket per eval step.
    assert len(report.accuracy_by_position) == 6
    for correct, total in report.accuracy_by_position:
        assert 0 <= correct <= total

    # the divergent-fact-type breakdown is non-negative and uses known fact classes.
    assert all(v >= 0 for v in report.divergence_by_fact_type.values())
    assert set(report.divergence_by_fact_type) <= {"file", "dir", "cwd", "env", "exit", "stdout"}


def test_run_diagnostics_deterministic():
    a = run_diagnostics(_tiny_config())
    b = run_diagnostics(_tiny_config())
    assert a.to_dict() == b.to_dict()
