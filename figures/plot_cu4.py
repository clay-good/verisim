"""Plot the CU4 figure (SPEC-22 §5, H96): the un-gameable safety gate.

Two panels from a :class:`~verisim.acd.adversarial_gate.CU4Result`:

  - **left -- gameable vs un-gameable.** The average-case missed-danger (over random attacks, blue)
    and the adversarial missed-danger (over the attacker's blind-spot arsenal, red) vs the
    consultation budget ρ. At ρ=0 the adversarial rate is **1.0** -- the free gate is gameable,
    far above the average-case rate -- and the shaded gap is the attacker's leverage. Verification
    collapses *both* to ≈0 at the cheap knee: the gate becomes un-gameable.
  - **right -- the false sense of security.** At ρ=0, across model fidelity φ: the average-case
    missed-danger falls as the model improves (blue), but the adversarial worst case stays pinned at
    **1.0 for every φ** (red) -- a more faithful model is no safer against an adversary. Only
    verification removes the worst case; faithfulness alone cannot.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.adversarial_gate import CU4Result


def plot_cu4(result: CU4Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    rhos = [c.rho for c in result.cells]
    avg = [c.avg_missed for c in result.cells]
    adv = [c.adversarial_missed for c in result.cells]
    ax_l.plot(rhos, adv, "-o", color="#d62728", lw=2.2, ms=8,
              label="adversarial (the attacker's blind-spot arsenal)")
    ax_l.plot(rhos, avg, "-s", color="#1f77b4", lw=2, ms=7,
              label="average-case (random attacks)")
    ax_l.fill_between(rhos, avg, adv, color="#d62728", alpha=0.08)
    cheap = next((c.rho for c in result.cells if c.adversarial_missed <= 0.05), None)
    if cheap is not None:
        ax_l.axvline(cheap, color="#2ca02c", ls=":", lw=1.5,
                     label=f"un-gameable at ρ = {cheap:g}")
    ax_l.set_xlabel("ρ  (oracle-consultation budget of the gate)")
    ax_l.set_ylabel("missed-danger rate\n(attacks the gate let through)")
    ax_l.set_title(
        f"a free gate is gameable (adversarial = 1.0); verification fixes it  (φ={result.phi})",
        fontsize=9.5,
    )
    ax_l.set_ylim(-0.03, 1.05)
    ax_l.legend(fontsize=8.5, loc="center right")

    phis = [p for p, _, _ in result.fidelity_independence]
    avg0 = [a for _, a, _ in result.fidelity_independence]
    adv0 = [d for _, _, d in result.fidelity_independence]
    ax_r.plot(phis, adv0, "-o", color="#d62728", lw=2.2, ms=9,
              label="adversarial missed-danger (worst case)")
    ax_r.plot(phis, avg0, "-s", color="#1f77b4", lw=2, ms=8,
              label="average-case missed-danger")
    ax_r.set_xlabel("model fidelity φ  (φ→1 = a more faithful model)")
    ax_r.set_ylabel("missed-danger rate at ρ=0 (no verification)")
    ax_r.set_title("a 'better' model is no safer against an adversary\n"
                   "(the worst case is fidelity-independent)")
    ax_r.set_ylim(-0.03, 1.05)
    ax_r.legend(fontsize=8.5, loc="center left")

    fig.suptitle(
        "CU4 / H96: a learned safety gate is GAMEABLE; verification makes it un-gameable — "
        "the oracle's value is worst-case robustness, not average faithfulness",
        fontsize=10, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
