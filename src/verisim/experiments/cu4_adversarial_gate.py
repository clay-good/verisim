"""Experiment CU4 -- the un-gameable safety gate (SPEC-22 §5, H96).

CU1-CU3 measured the gate against a *random* world. CU4 asks the question a security threat model
demands: *is the gate gameable by an attacker who knows the deployed model?* The answer is the one
that makes verification matter for cyber: a free (learned-only) gate is **fully gameable**, and the
oracle-in-the-loop makes it **un-gameable** at the cheap knee.

The run ([`acd/adversarial_gate.py`](../acd/adversarial_gate.py)): a battery of attacks (plans that
truly write a protected prefix); the gate previews each through the deployed model (ρ-grounded) and
allows it iff the preview shows no protected write. The **average-case** missed-danger is over
random attacks; the **adversarial** missed-danger is over the attacker's arsenal — the attacks it
previews as safe (its blind spots), which the attacker fires by choice.

Two findings (H96): (1) a free gate's adversarial missed-danger is **1.0** (every crafted attack
succeeds), far above its average rate — and verification collapses *both* to ≈0 at the cheap knee
(un-gameable); (2) the adversarial worst case at ρ=0 is **1.0 for *any* model fidelity** — a more
faithful model lowers the *average* miss but never the *adversarial* one, so average faithfulness
is a false sense of security; only verification removes the worst case. CPU, torch-free, seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.adversarial_gate import (
    CU4Config,
    cu4_verdict,
    run_cu4,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU4 -- the un-gameable safety gate (SPEC-22 H96)."
    )
    parser.add_argument("--out", type=str, default="figures/cu4_adversarial_gate.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU4Config.smoke() if args.smoke else CU4Config()
    result = run_cu4(config)
    verdict = cu4_verdict(result)

    print(f"CU4 / H96 -- the un-gameable safety gate (φ={result.phi}, "
          f"{result.n_attacks}/{result.n_plans} attacks, {result.mean_blind_fraction:.0%} are "
          f"free-model blind spots):")
    print(f"  {'rho':>5} {'avg-case missed':>16} {'adversarial missed':>20}")
    for c in result.cells:
        print(f"  {c.rho:>5.2f} {c.avg_missed:>16.3f} {c.adversarial_missed:>20.3f}")
    print("\n  fidelity-independence of the worst case (at ρ=0 — a 'better' model is no safer):")
    print(f"  {'φ':>5} {'avg-case missed':>16} {'adversarial missed':>20}")
    for phi, avg0, adv0 in result.fidelity_independence:
        print(f"  {phi:>5.2f} {avg0:>16.3f} {adv0:>20.3f}")

    print(f"\n  a free gate (ρ=0) is gameable (adversarial missed ≈ 1.0): "
          f"{verdict['free_gate_gameable']} ({verdict['free_adversarial_missed']:.2f} adversarial "
          f"vs {verdict['free_avg_missed']:.2f} average)")
    print(f"  verification makes it un-gameable: {verdict['verification_un_gameable']}  "
          f"(worst case ≈0 by ρ = {verdict['un_gameable_rho']:.2f})")
    print(f"  the worst case is fidelity-independent (≈1.0 for every φ): "
          f"{verdict['worst_case_fidelity_independent']} — average-case faithfulness is a false "
          f"sense of security")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu4 import plot_cu4

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu4(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
