"""SPEC-24 (H165/H166) -- the graded guarantee: what the coverage certificate actually proves.

SPEC-23's certificate says "no oracle-confirmed hole in the sampled space." That is a fuzzer's
output, not a coverage class -- and you cannot sell it about *someone else's* guardrail without
saying
what it is worth. This module supplies the two tiers that, together, are the honest whole:

  - **exhaustive bounded sub-certificate (a theorem for a fragment).** The compositional action
  space
    is exponential overall (``|MECH|^|ATOMS| x |VERB|`` ~ 17.9M) but *depth-bounded* it is finite
    and
    small: depth <= k actions = ``(sum_{j<=k} C(|ATOMS|,j) (|MECH|-1)^j) x |VERB|`` -- 402 /
    11,292 /
    171,012 at k=1/2/3. Enumerating it all (no sampling, no seed) makes "no in-contract hole at
    depth <= k" a **proof over a complete fragment**. :func:`depth_bounded_count` is the closed form
    the enumerator is checked against.

  - **statistical residual bound (the unbounded tail).** Beyond depth k the space is too large to
    enumerate, so the sampled tail carries a quantified residual. :func:`rule_of_three_upper` is the
    exact one-sided Clopper-Pearson upper bound for *zero* observed holes in ``n`` draws
    (``-ln(delta)/n``, ~``3/n`` at 95%); :func:`wilson_upper` is the closed-form score-interval
    upper
    bound when holes *are* observed; :func:`good_turing_missing_mass` estimates the probability the
    next draw is a *new* hole type. Together they turn "none found in N" into "residual in-contract
    hole rate < epsilon at confidence 1-delta."

The :class:`Guarantee` block records both tiers plus the **modeled action algebra** and its size, so
the certificate's claim is never read wider than it is: it is complete over *that algebra*, to that
depth, with that residual -- not over all bash. Growing the algebra is the RA25 enumeration
treadmill.

Torch-free, standard library only, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb


def depth_bounded_count(n_atoms: int, n_mech: int, k: int, n_verb: int = 1) -> int:
    """The exact size of the direct-redirect composition space at depth <= ``k`` (the closed form
    the
    exhaustive enumerator is verified against). ``depth`` = number of atoms encoded by a non-literal
    mechanism, so there are ``C(n_atoms, j)`` ways to pick which ``j`` atoms are non-literal and
    ``(n_mech-1)^j`` mechanism assignments for them."""
    comps = sum(comb(n_atoms, j) * (n_mech - 1) ** j for j in range(min(k, n_atoms) + 1))
    return int(comps * n_verb)


# --- residual estimators (the unbounded tail) -----------------------------------------------------


def rule_of_three_upper(n: int, delta: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson upper bound on a rate given **zero** events in ``n`` trials:
    ``1 - delta^(1/n)``, which is ``~= -ln(delta)/n`` and ``~= 3/n`` at the 95% level. The honest
    meaning of "we sampled ``n`` and found no hole.\""""
    if n <= 0:
        return 1.0
    return float(1.0 - delta ** (1.0 / n))


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Closed-form Wilson score-interval **upper** bound on the true rate given ``k`` events in
    ``n``
    trials (no special functions, so dependency-free). ``z=1.96`` is the 95% two-sided level."""
    if n <= 0:
        return 1.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4 * n * n))
    return min(1.0, center + margin)


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Closed-form Wilson score-interval **lower** bound (companion to :func:`wilson_upper`)."""
    if n <= 0:
        return 0.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4 * n * n))
    return max(0.0, center - margin)


def good_turing_missing_mass(n_singletons: int, n_sampled: int) -> float:
    """Good-Turing estimate of the missing mass: the probability the *next* sampled realizing action
    is a hole *type* not yet seen, estimated as (# hole types seen exactly once) / n."""
    if n_sampled <= 0:
        return 1.0
    return n_singletons / n_sampled


def residual_epsilon(k_holes: int, n_sampled: int, delta: float = 0.05) -> float:
    """The certificate's residual: an upper bound on the in-contract hole rate over the sampled
    tail.
    Uses the exact zero-event bound when no hole was sampled, the Wilson upper bound otherwise."""
    if k_holes == 0:
        return rule_of_three_upper(n_sampled, delta)
    return wilson_upper(k_holes, n_sampled, z=_z_for(delta))


def _z_for(delta: float) -> float:
    """The two-sided z for a few common confidence levels (no scipy)."""
    return {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}.get(round(delta, 2), 1.96)


# --- the guarantee block --------------------------------------------------------------------------


@dataclass
class Guarantee:
    """What a certificate proves: complete over ``algebra`` (size ``algebra_size``), exhaustively
    sound to ``exhaustive_depth`` (``n_exhaustive`` actions enumerated, no sampling), and beyond
    that
    depth bounded to a residual in-contract hole rate < ``residual_epsilon`` at confidence
    ``1-residual_delta`` over ``n_sampled`` draws. ``external`` names a third-party auditee if
    any."""

    algebra: str
    algebra_size: int
    exhaustive_depth: int           # k proven exhaustively; -1 == no exhaustive pass
    n_exhaustive: int
    residual_epsilon: float
    residual_delta: float
    n_sampled: int
    missing_mass: float
    oracle: str
    external: str = ""

    def grade(self) -> str:
        """A one-line human summary of the coverage class this certificate carries."""
        ex = (f"PROVEN<=depth{self.exhaustive_depth} (n={self.n_exhaustive})"
              if self.exhaustive_depth >= 0 else "no exhaustive tier")
        tail = (f"residual<{self.residual_epsilon:.4f} @ conf {1 - self.residual_delta:.2f} "
                f"(n={self.n_sampled})")
        ext = f" vs external={self.external}" if self.external else ""
        return f"{ex} + {tail} over {self.algebra}{ext}"

    def meets(self, *, depth: int | None = None, epsilon: float | None = None) -> bool:
        """Does this guarantee meet a declared CI floor? (used by the ``--require-*`` gate)."""
        ok_depth = depth is None or self.exhaustive_depth >= depth
        ok_epsilon = epsilon is None or self.residual_epsilon <= epsilon
        return ok_depth and ok_epsilon
