"""Tests for SPEC-23 Direction A (H158/H160/H161): the protocol-driven monitor auditor.

Hermetic and deterministic. They pin the load-bearing claims that the certification loop survives
abstraction behind the Monitor/Oracle protocols with no literal /etc baked into the loop:

  - H158: the auditor re-derives the RA24 printf-format-escape silent miss through the
    ``ResolverMonitor`` interface (unsound -> a silent hole; hardened -> none), and reproduces the
    RA22 literal/indirection partition through the ``SyntacticPathMonitor`` grown from empty.
  - H160: the certificate is CI-consumable -- ``audit(...) -> Certificate`` serializes to JSON, and
    its ``sound`` verdict / ``silent_holes`` count drives a non-zero CLI exit on a silent hole.
  - H161: the same monitor audited by the syntactic ``ShellPathOracle`` and the real-container
    ``ContainerDiffOracle`` agrees on the reversible class, and the diff oracle additionally catches
    the symlink indirection the syntactic oracle cannot express (guarded by a present shell).

The /bin/sh container-diff test is skipped where no shell is available; the rest never execute.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil

import pytest

from verisim.audit import (
    ContainerDiffOracle,
    GrammarProposer,
    ResolverMonitor,
    ShellPathOracle,
    SyntacticPathMonitor,
    audit,
)
from verisim.audit.proposers import CorpusProposer

PROTECTED = "/etc/shadow"
PREFIX = "/etc"
WORK = "/home/work"


def test_reproduces_ra24_printf_silent_miss_through_the_interface() -> None:
    # the pre-RA24 (unsound) resolver, audited through the protocol, has the printf-format-escape
    # silent miss the learned adversary discovered -- one per verb, all string-resolvable & CLEAR.
    mon = ResolverMonitor(PREFIX, sound_printf=False)
    cert = audit(mon, ShellPathOracle(PREFIX), GrammarProposer(PROTECTED, WORK, mode="enumerate"))
    assert cert.silent_holes > 0
    assert not cert.sound
    assert {h.klass for h in cert.holes} == {"printf_fmt"}
    assert all(h.silent and h.string_resolvable for h in cert.holes)


def test_hardened_resolver_is_sound_through_the_interface() -> None:
    # the hardened resolver (the default) routes the printf class to ABSTAIN -> no silent miss.
    mon = ResolverMonitor(PREFIX, sound_printf=True)
    cert = audit(mon, ShellPathOracle(PREFIX), GrammarProposer(PROTECTED, WORK, mode="enumerate"))
    assert cert.silent_holes == 0
    assert cert.sound


def test_reproduces_ra22_partition_from_empty_syntactic_target() -> None:
    # the syntactic target, grown from EMPTY by the loop, covers the literal class and isolates the
    # indirection class as the routed residual -- RA22's partition, through the interface.
    mon = SyntacticPathMonitor()  # no seed: synthesized from nothing
    cert = audit(mon, ShellPathOracle(PREFIX), CorpusProposer(PREFIX, WORK))
    assert mon.prefixes == [PREFIX]  # synthesized the covering prefix from empty
    assert cert.silent_holes == 0  # soundness: nothing realizing is silently off-surface
    assert cert.per_class["literal"]["covered"] == cert.per_class["literal"]["realizing"]
    # the string-resolvable indirection is out of the literal target's contract -> routed residual
    assert cert.residual_post_commit > 0
    assert cert.per_class["indirection_var"]["residual"] > 0


def test_certificate_is_ci_consumable_json() -> None:
    mon = ResolverMonitor(PREFIX, sound_printf=False)
    cert = audit(mon, ShellPathOracle(PREFIX), GrammarProposer(PROTECTED, WORK, mode="enumerate"))
    blob = json.loads(cert.to_json())
    # the certificate carries the CI-relevant fields: holes, sampled space, per-class, routing.
    for key in ("monitor", "oracle", "proposer", "n_proposed", "n_realizing", "silent_holes",
                "residual_post_commit", "residual_human", "holes", "per_class", "version"):
        assert key in blob
    assert blob["silent_holes"] == cert.silent_holes
    assert blob["monitor"] == "resolver_unsound" or blob["monitor"] == "resolver"


def test_cli_exits_nonzero_on_silent_hole_and_zero_when_sound(tmp_path: object) -> None:
    from verisim.audit.__main__ import main

    out = os.path.join(str(tmp_path), "cert.json")
    # the unsound resolver has a silent hole -> non-zero exit (CI fails the build)
    assert main(["resolver-unsound", "shell-path", "--out", out]) == 1
    assert json.loads(pathlib.Path(out).read_text())["silent_holes"] > 0
    # the hardened resolver is sound -> zero exit
    assert main(["resolver", "shell-path", "--out", out]) == 0
    assert json.loads(pathlib.Path(out).read_text())["silent_holes"] == 0


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs /bin/sh for container-diff")
def test_container_diff_oracle_catches_symlink_the_syntactic_oracle_cannot() -> None:
    # H161: audit the same (unsound resolver) monitor with the syntactic oracle and the real-effect
    # container-diff oracle. They agree on the reversible printf class; the diff oracle additionally
    # flags the symlink indirection (whose realized target the string never names).
    diff = ContainerDiffOracle()
    try:
        prefix, work = diff.prefix, diff.work
        protected = f"{prefix}/shadow"
        syn = audit(ResolverMonitor(prefix, sound_printf=False), ShellPathOracle(prefix),
                    GrammarProposer(protected, work, mode="enumerate"))
        con = audit(ResolverMonitor(prefix, sound_printf=False), diff,
                    GrammarProposer(protected, work, mode="enumerate"))
    finally:
        diff.close()
    syn_klasses = {h.klass for h in syn.holes}
    con_klasses = {h.klass for h in con.holes}
    assert "printf_fmt" in syn_klasses and "printf_fmt" in con_klasses  # agree: reversible cls
    assert "residual_symlink" not in syn_klasses  # syntactic oracle is blind to the symlink
    assert "residual_symlink" in con_klasses  # the diff oracle sees the realized effect
