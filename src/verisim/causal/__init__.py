"""Causal / counterfactual world models: the oracle as an exact Structural Causal Model (SPEC-17).

A deterministic, resettable, seedable oracle is not *like* an SCM -- it *is* one, exactly and for
free
(SPEC-17 §2): the structural equations ``F`` are ``oracle.step``, the exogenous noise ``U`` is the
seed/clock the rollout consumes, and **abduction is reset+replay** -- recovering the exact ``U``
behind
a factual state is an ``O(1)`` lookup, not the intractable inference it is in an oracle-free SCM.
From
that one identification, all three rungs of Pearl's ladder are executable exactly:

  - rung 1 (observe): roll the oracle on the on-policy action distribution;
  - rung 2 (intervene, ``do(a')``): ``oracle.step(s, a')`` -- the shipped counterfactual_branches;
  - rung 3 (counterfactual): abduct ``Û`` from ``(seed, t)``, intervene, re-run ``F`` with
    the *same* ``Û`` -- the true "what the factual rollout would have become under the same noise
    had
    one action differed" (:func:`~verisim.causal.scm.rung3_counterfactual`).

This module is the world-generic SCM machinery; the CX experiments
(:mod:`verisim.experiments.cx0` …) measure the do-calculus reading of H5 on top of it.
"""

from .scm import (
    Intervention,
    abduct_and_replay,
    abduction_exact,
    downstream_amplification,
    rung2_branch,
    rung3_counterfactual,
)

__all__ = [
    "Intervention",
    "abduct_and_replay",
    "abduction_exact",
    "downstream_amplification",
    "rung2_branch",
    "rung3_counterfactual",
]
