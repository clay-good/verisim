"""Tests for SPEC-22 RA4 (H136): the command-agnostic gate.

Hermetic string/grammar analysis over labeled corpora (no command execution). They pin the three
findings: the command-agnostic target catches every explicit-reference attack syntax (including ones
the v0 grammar cannot parse), it is silent on real benign commands it never saw, and -- honestly --
the syntactic pre-commit target is evaded by indirection, which is why the principled gate routes
irreversible dangers through CU27's exact post-commit fs-diff.
"""

from __future__ import annotations

from verisim.realagent.command_agnostic import (
    ATTACK_SYNTAXES,
    BENIGN_CORPUS,
    EVASIONS,
    command_targets_protected,
    cu_ra4_verdict,
    grammar_can_parse,
    run_ra4,
)


def test_target_catches_every_explicit_reference_attack() -> None:
    for c in ATTACK_SYNTAXES:
        assert command_targets_protected(c.command) is True, c.name


def test_grammar_gate_is_structurally_blind_to_most_attack_syntaxes() -> None:
    # the v0 grammar cannot even parse redirects / tee / dd / python / sed -i / rm -f / ...
    blind = [c for c in ATTACK_SYNTAXES if not grammar_can_parse(c.command)]
    assert len(blind) > len(ATTACK_SYNTAXES) / 2  # a grammar gate sees less than half
    # every attack the grammar gate is blind to, the command-agnostic target still catches
    for c in blind:
        assert command_targets_protected(c.command) is True, c.name


def test_no_false_blocks_on_unseen_benign_commands() -> None:
    for c in BENIGN_CORPUS:
        assert command_targets_protected(c.command) is False, c.name


def test_pre_commit_target_is_evaded_by_indirection_honest_negative() -> None:
    # the open edge: a path built without the literal prefix (or via a symlink) slips a scan
    for c in EVASIONS:
        assert command_targets_protected(c.command) is False, c.name


def test_target_is_prefix_relative_not_substring() -> None:
    # a sibling path that merely shares a prefix string must NOT fire (no false positive)
    assert command_targets_protected("rm /etcetera/notes", "/etc") is False
    assert command_targets_protected("rm /etc/shadow", "/etc") is True
    assert command_targets_protected("rm /etc", "/etc") is True


def test_verdict_flags() -> None:
    v = cu_ra4_verdict(run_ra4())
    assert v["target_catches_all_explicit_attacks"] is True
    assert v["grammar_gate_is_blind"] is True
    assert v["no_false_blocks_on_benign"] is True
    assert v["pre_commit_target_has_evasions"] is True
    assert v["post_commit_diff_is_exact"] is True
