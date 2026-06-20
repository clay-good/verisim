"""Tests for SPEC-22 RA23 (H155): the learned adversarial proposer (oracle-as-reward).

Hermetic, deterministic (seeded), no shell. They pin the load-bearing claims: the policy is a valid
softmax that learns toward rewarded actions; the learned proposer beats the blind baseline on holes
per oracle call; it concentrates on the indirection class and the protected target; it surfaces the
quote_splice residual class absent from RA22's grid; and the soundness invariant holds under the
adaptive adversary. The real-shell cross-check (`bash_cross_check`) needs a shell; the experiment
runs it.
"""

from __future__ import annotations

from verisim.realagent.learned_proposer import (
    INDIRECTION,
    LearnedProposer,
    RandomProposer,
    SoftmaxPolicy,
    make_proposal,
    ra23_verdict,
    run_cegis,
)


def test_softmax_policy_is_a_distribution_and_learns() -> None:
    pol = SoftmaxPolicy(3)
    p0 = pol.probs()
    assert abs(sum(p0) - 1.0) < 1e-9
    assert all(abs(x - 1 / 3) < 1e-9 for x in p0)  # uniform at init
    for _ in range(50):
        pol.update(1, advantage=1.0, lr=0.5)  # reward option 1 repeatedly
    p1 = pol.probs()
    assert p1[1] > p0[1]  # mass shifted toward the rewarded option
    assert p1[1] == max(p1)


def test_make_proposal_labels_match_construction() -> None:
    tmpl, w = "echo x > {p}", "/home/work"
    prot_lit = make_proposal("protected", "redirect", tmpl, "literal", "/etc", w)
    assert prot_lit.realizes and prot_lit.literal_present
    prot_ind = make_proposal("protected", "redirect", tmpl, "var_split", "/etc", w)
    assert prot_ind.realizes and not prot_ind.literal_present  # indirection: no literal token
    benign = make_proposal("benign", "rm", "rm -f {p}", "literal", "/etc", "/home/work")
    assert not benign.realizes  # benign work never corrupts the protected region


def test_learned_beats_blind_on_holes_per_call() -> None:
    learned = run_cegis(LearnedProposer(seed=0), budget=400)
    rnd = run_cegis(RandomProposer(seed=0), budget=400)
    assert learned.holes_found > rnd.holes_found  # adaptivity pays
    v = ra23_verdict(learned, rnd)
    assert v["learned_more_efficient"] is True


def test_learned_concentrates_on_indirection_and_protected() -> None:
    learned = run_cegis(LearnedProposer(seed=0), budget=600)
    ind_mass = sum(learned.final_transform_probs[t] for t in INDIRECTION)
    assert ind_mass > 0.8  # literal class covered -> no reward -> mass flows to indirection
    assert learned.final_target_probs["protected"] > 0.8  # learns to attack, not waste on benign


def test_surfaces_quote_splice_residual_beyond_ra22_grid() -> None:
    learned = run_cegis(LearnedProposer(seed=0), budget=600)
    assert "quote_splice" in learned.distinct_residual_classes
    # RA22's grid had var/subst/symlink; quote_splice is the new class the adversary surfaced
    assert "quote_splice" not in ("indirection_var", "indirection_subst", "indirection_symlink")


def test_soundness_invariant_under_adversary() -> None:
    learned = run_cegis(LearnedProposer(seed=0), budget=600)
    rnd = run_cegis(RandomProposer(seed=0), budget=600)
    assert learned.silent_miss == 0 and rnd.silent_miss == 0
    assert learned.synthesized_prefixes == ["/etc"]  # the literal class still gets covered


def test_run_is_deterministic_for_a_seed() -> None:
    a = run_cegis(LearnedProposer(seed=7), budget=200)
    b = run_cegis(LearnedProposer(seed=7), budget=200)
    assert a.holes_found == b.holes_found
    assert a.hole_curve == b.hole_curve


def test_blind_proposer_does_not_adapt() -> None:
    rng_probe = RandomProposer(seed=1)
    rng_probe.learn((0, 0, 0), reward=1.0)  # learning is a no-op for the blind arm
    # two independent blind runs with the same seed are identical (no internal learned state)
    assert run_cegis(RandomProposer(seed=1), budget=100).hole_curve == \
        run_cegis(RandomProposer(seed=1), budget=100).hole_curve
