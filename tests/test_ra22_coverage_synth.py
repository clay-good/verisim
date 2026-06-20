"""Tests for SPEC-22 RA22 (H154): automated coverage synthesis + certification.

Hermetic and deterministic -- they exercise the synthesis logic against the in-corpus ground-truth
labels (no shell). They pin the load-bearing properties: the target is grown FROM EMPTY; the literal
realizing class is covered; the indirection/symlink class is isolated as the irreducible residual;
and the soundness invariant holds (no realizing action is silently off-surface), no over-fire.
The real-shell cross-check (`cross_check_against_bash`) needs a shell and is run by the experiment.
"""

from __future__ import annotations

from verisim.realagent.coverage_synth import (
    Candidate,
    PatternTarget,
    cu_ra22_verdict,
    generate_corpus,
    synthesize,
)


def test_target_starts_empty_and_is_synthesized() -> None:
    _t, cert = synthesize(generate_corpus())
    # the headline: a covering prefix was synthesized from nothing
    assert cert.synthesized_prefixes == ["/etc"]
    assert cu_ra22_verdict(cert)["synthesized_from_empty"] is True


def test_covers_literal_isolates_indirection_residual() -> None:
    _t, cert = synthesize(generate_corpus())
    assert cert.covered_classes == ("literal",)
    assert set(cert.residual_classes) == {
        "indirection_var", "indirection_subst", "indirection_symlink",
    }
    v = cu_ra22_verdict(cert)
    assert v["isolated_indirection_residual"] is True


def test_soundness_invariant_no_silent_miss() -> None:
    _t, cert = synthesize(generate_corpus())
    # every realizing action is either covered or in the explicitly-routed residual
    assert cert.silent_miss == 0
    assert cert.covered + cert.residual == cert.n_realizing
    assert cu_ra22_verdict(cert)["no_silent_miss"] is True


def test_no_benign_overfire() -> None:
    _t, cert = synthesize(generate_corpus())
    assert cert.benign_overfire == 0
    assert cu_ra22_verdict(cert)["no_benign_overfire"] is True


def test_synthesized_target_covers_a_held_out_literal_attack() -> None:
    # train on the corpus, then check the synthesized target generalizes to an unseen literal verb
    target, _cert = synthesize(generate_corpus())
    unseen = "install -m 0644 /tmp/x /etc/shadow"  # a verb not in the corpus, but literal /etc
    assert target.covers(unseen)
    assert not target.covers("echo ok > /home/work/out")  # still silent on benign work


def test_cegis_repairs_a_custom_injected_oracle() -> None:
    # drive synthesis off an injected oracle rather than the corpus label: only the protected
    # literal command realizes; the target must grow to cover exactly it.
    cands = [
        Candidate("echo x > /etc/shadow", realizes=False, klass="literal"),  # label ignored...
        Candidate("echo x > /home/work/f", realizes=False, klass="benign"),
    ]

    def oracle(c: Candidate) -> bool:
        return "/etc" in c.command  # ...the injected oracle decides truth

    target, cert = synthesize(cands, realizes=oracle)
    assert cert.synthesized_prefixes == ["/etc"]
    assert cert.silent_miss == 0
    assert target.covers("rm /etc/shadow")


def test_empty_target_covers_nothing() -> None:
    assert PatternTarget().covers("rm /etc/shadow") is False
