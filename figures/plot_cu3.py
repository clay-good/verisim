"""Plot the CU3 figure (SPEC-22 §5, H95): the certified safety gate.

One panel from a :class:`~verisim.acd.certified_gate.CU3Result`: the certificate's **validity** and
its **cost** versus the consultation budget ρ. The certified missed-danger rate (blue) stays at or
below the target α (the dashed line) at every ρ -- the distribution-free guarantee holds. The
false-block rate (red) -- the cost of the guarantee, the safe plans the gate wrongly aborts --
**collapses with faithfulness**: from ≈ 1 at ρ=0 (a drifting preview can only be safe by aborting
everything, useless) to ≈ 0 by ρ≈0.2 (the oracle-grounded preview certifies the same guarantee while
allowing the safe plans). Any model can be made safe by being useless; only a faithful one is safe
*and* useful, and ρ buys that ≈ free.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.certified_gate import CU3Result


def plot_cu3(result: CU3Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.4))
    rhos = [c.rho for c in result.cells]

    ax.plot(rhos, [c.false_block for c in result.cells], "-o", color="#d62728", lw=2.2, ms=8,
            label="false-block rate (the cost of the guarantee)")
    ax.plot(rhos, [c.missed_danger for c in result.cells], "-s", color="#1f77b4", lw=2, ms=7,
            label="certified missed-danger rate (the guarantee)")
    ax.axhline(result.alpha, color="#1f77b4", ls=":", lw=1.5,
               label=f"target α = {result.alpha} (the certificate holds below this)")

    cheap = next((c.rho for c in result.cells if c.false_block <= 0.05), None)
    if cheap is not None:
        ax.axvline(cheap, color="#2ca02c", ls=":", lw=1.5,
                   label=f"guarantee ≈ free at ρ = {cheap:g}")

    ax.set_xlabel("ρ  (oracle-consultation budget of the preview)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        "the safety certificate holds at every ρ; its cost collapses with faithfulness\n"
        "(any model is safe by aborting everything — only a faithful one is safe AND useful)"
    )
    ax.legend(fontsize=9, loc="center right")

    fig.suptitle(
        "CU3 / H95: an agent can CERTIFY P(missed danger) ≤ α for free with the oracle",
        fontsize=11, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
