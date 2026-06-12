"""Plot the CU2-net figure (SPEC-22 §5): the cross-world exfiltration safety gate.

Two panels from a :class:`~verisim.experiments.cu2_net_gate.CU2NetResult`:

  - **left -- missed danger, free vs oracle.** On the network world, an exfiltration guardrail (no
    flow to a protected server) gated by a free (unverified) network preview vs the oracle: the
    free preview misses real exfil dangers (the model drifts on which flows establish), the oracle
    misses none -- the boundary law reproduced cross-world.
  - **right -- the cheap safe gate.** The missed-danger rate vs ρ: the oracle-in-the-loop drives it
    to zero at a small consultation budget (the UA10 network knee, here on agent safety).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.cu2_net_gate import CU2NetResult, knee_rho


def plot_cu2_net(result: CU2NetResult, path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax_l.bar([0], [result.free.missed_danger_rate], 0.5, color="#d62728",
             label="free preview (unverified net M_θ)")
    ax_l.bar([1], [result.oracle.missed_danger_rate], 0.5, color="#1f77b4",
             label="oracle preview (verified)")
    ax_l.text(0, result.free.missed_danger_rate + 0.02,
              f"{result.free.missed_dangers} exfil\nplans run", ha="center", fontsize=8,
              color="#d62728")
    ax_l.set_xticks([0, 1])
    ax_l.set_xticklabels(["free", "oracle"])
    ax_l.set_ylabel("missed-danger rate\n(exfil plans the agent executed)")
    ax_l.set_title("the exfiltration gate, cross-world (network flows)")
    ax_l.set_ylim(0, 1.05)
    ax_l.legend(fontsize=8, loc="upper right")

    knee = sorted(result.knee, key=lambda z: z[0])
    rhos = [r for r, _ in knee]
    mdr = [o.missed_danger_rate for _, o in knee]
    ax_r.plot(rhos, mdr, "-o", color="#d62728")
    safe_rho = knee_rho(result)
    ax_r.axvline(safe_rho, color="#2ca02c", ls=":", label=f"missed-danger = 0 at ρ = {safe_rho:g}")
    ax_r.axhline(0, color="#888", lw=0.8)
    ax_r.set_xlabel("ρ  (oracle-consultation budget of the preview)")
    ax_r.set_ylabel("missed-danger rate")
    ax_r.set_title("verification buys a safe exfil gate cheaply (the knee)")
    ax_r.set_ylim(bottom=-0.03)
    ax_r.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "CU2-net / H93 cross-world: a network defender needs a verified world model to gate "
        "exfiltration safely — and the oracle buys it cheaply"
    )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
