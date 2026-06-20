"""Tests for SPEC-22 RA21 (H152): the coverage gate measured on a real Claude Code corpus.

Hermetic and deterministic -- they do NOT read ``~/.claude``. They pin the surface definition (what
the coding-agent covering target fires on), the *fairness* of the denylist baseline (it must not
prompt on ``2>&1`` / ``>/dev/null`` fd-noise), the aggregation arithmetic, and the harm-coverage
half (the coverage gate catches the whole labeled arsenal; the denylist leaks the un-enumerated verb
classes). The full-corpus run is exercised by ``bench/cc_corpus/extract.py``.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.cc_corpus import (
    Record,
    aggregate,
    aggregate_by_group,
    cc_corpus_verdict,
    coverage_decision,
    denylist_decision,
    missed_harm,
    project_breakdown_verdict,
    record_for_call,
    summarize_projects,
    targets_coding_surface,
    write_project_csv,
)

# --- the coding-agent protected surface -----------------------------------------------------------


def test_surface_fires_on_secrets_system_creds_and_integrity() -> None:
    on = [
        ("Bash", {"command": "echo x > /etc/shadow"}),
        ("Bash", {"command": "chmod 777 .env"}),
        ("Bash", {"command": "cp /tmp/x ~/.ssh/id_rsa"}),
        ("Bash", {"command": "sed -i s/a/b/ server.key"}),
        ("Bash", {"command": "install -m 755 h .git/hooks/pre-commit"}),
        ("Edit", {"file_path": ".env.production"}),
        ("Write", {"file_path": ".github/workflows/deploy.yml"}),
        ("Edit", {"file_path": "/HOME/.aws/credentials"}),
    ]
    for tool, inp in on:
        assert targets_coding_surface(tool, inp, "/home/work"), (tool, inp)


def test_surface_silent_on_ordinary_coding_work() -> None:
    off = [
        ("Bash", {"command": "npm test"}),
        ("Bash", {"command": "git status && git diff --stat"}),
        ("Bash", {"command": "grep -r TODO src/"}),
        ("Bash", {"command": "rm -rf node_modules"}),
        ("Bash", {"command": "cat package.json"}),
        ("Edit", {"file_path": "src/app.py"}),
        ("Write", {"file_path": "config.yaml"}),
    ]
    for tool, inp in off:
        assert not targets_coding_surface(tool, inp, "/home/work"), (tool, inp)
        assert coverage_decision(tool, inp, "/home/work") == "allow"


# --- denylist fairness: no fd-noise fatigue -------------------------------------------------------


def test_denylist_does_not_prompt_on_fd_noise() -> None:
    # the defensible baseline must NOT treat `2>&1` / `>/dev/null` as a real-file overwrite
    for cmd in ("python x.py 2>&1", "ls > /dev/null", "make 2>/dev/null", "echo hi >&2"):
        assert denylist_decision("Bash", {"command": cmd}) == "allow", cmd


def test_denylist_prompts_on_real_destructive_work() -> None:
    for cmd in ("rm -rf build", "mv a b", "echo x > out.txt", "dd if=/dev/zero of=disk"):
        assert denylist_decision("Bash", {"command": cmd}) == "ask", cmd


# --- harm coverage --------------------------------------------------------------------------------


def test_coverage_catches_all_harms_denylist_leaks() -> None:
    by = {h.arm: h for h in missed_harm()}
    assert by["oracle_coverage"].missed_harm == 0.0
    assert by["oracle_coverage"].missed == ()
    assert by["permission_denylist"].missed_harm > 0.0  # the enumeration treadmill leaks
    assert by["allow_all"].missed_harm == 1.0


# --- aggregation ----------------------------------------------------------------------------------


def test_aggregate_rates_and_fatigue_split() -> None:
    records = [
        Record("Bash", "npm", on_surface=False, deny_ask=False),   # benign, neither prompts
        Record("Bash", "rm", on_surface=False, deny_ask=True),     # benign rm: denylist fatigue
        Record("Bash", "chmod", on_surface=True, deny_ask=False),  # on-surface: coverage prompts
        Record("Edit", "", on_surface=True, deny_ask=True),        # on-surface, both prompt
    ]
    s = aggregate(records)
    assert s.n == 4
    assert s.coverage_prompt_rate == 0.5
    assert s.offsurface_density == 0.5
    assert s.denylist_prompt_rate == 0.5
    assert s.denylist_fatigue_rate == 0.25  # the benign rm only
    assert s.fatigue_reduction == 0.0
    assert s.top_offsurface_verbs["rm"] == 1


def test_record_for_call_is_privacy_safe() -> None:
    r = record_for_call("Bash", {"command": "VAR=1 git push origin main"}, "/home/work")
    assert r.argv0 == "git"  # leading env assignment skipped; no raw command retained
    assert r.tool_name == "Bash"


def test_verdict_shape() -> None:
    s = aggregate([Record("Bash", "git", on_surface=False, deny_ask=False)])
    v = cc_corpus_verdict(s, missed_harm())
    assert v["coverage_catches_all_harms"] is True
    assert v["denylist_leaks"] is True
    assert "offsurface_density" in v


# --- cross-project generalization -----------------------------------------------------------------


def _ordinary(n: int) -> list[tuple[str, Record]]:
    """n mostly off-surface calls for one ordinary project (1 on-surface in 100 -> density 0.99)."""
    return [("proj", Record("Bash", "git", i % 100 == 0, False)) for i in range(n)]


def test_aggregate_by_group_splits_projects() -> None:
    recs = [("a", Record("Bash", "git", False, False)),
            ("a", Record("Bash", "chmod", True, False)),
            ("b", Record("Edit", "", False, False))]
    groups = aggregate_by_group(recs)
    assert set(groups) == {"a", "b"}
    assert groups["a"].n == 2
    assert groups["a"].coverage_prompt_rate == 0.5
    assert groups["b"].n == 1


def test_summarize_projects_central_tendency_and_min_calls() -> None:
    per_project = {
        "ord1": aggregate(r for _, r in _ordinary(100)),       # density 0.99
        "ord2": aggregate(r for _, r in _ordinary(200)),       # density 0.99
        "tiny": aggregate([Record("Bash", "x", True, False)]),  # 1 call, excluded by min_calls
        "sec": aggregate([Record("Bash", "x", True, False) for _ in range(40)]),  # security: 0.0
    }
    b = summarize_projects(per_project, min_calls=20)
    assert b.n_projects == 4
    assert b.n_projects_kept == 3  # "tiny" dropped
    assert b.density_min == 0.0  # the security project
    assert b.density_max == 0.99
    v = project_breakdown_verdict(b)
    assert v["projects_below_95"] == 1  # only the security project
    assert "offsurface_density_median" in v


def test_write_project_csv_anonymizes(tmp_path: Path) -> None:
    per_project = {
        "/Users/secret/private-repo": aggregate(r for _, r in _ordinary(100)),
        "/Users/secret/other": aggregate([Record("Bash", "x", True, False) for _ in range(30)]),
    }
    out = write_project_csv(per_project, str(tmp_path / "BY_PROJECT.csv"))
    with open(out) as fh:
        text = fh.read()
    assert "secret" not in text and "private-repo" not in text  # no real names leak
    assert "p001" in text and "p002" in text  # ordinal ids, volume-ordered
