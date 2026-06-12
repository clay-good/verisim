"""Plot the CU2-threats figure (SPEC-22 §5): the safety gate across a cyber threat spectrum.

Two panels from a :class:`~verisim.experiments.cu2_threats.CU2ThreatsResult`:

  - **left -- the threat spectrum.** The free (unverified) preview's missed-danger rate per threat,
    ordered structure→content: service kill (~0, the model is faithful on the process tree) <
    privilege escalation (the model drifts moderately on uid) < credential tampering (the model
    drifts most on file content). The oracle preview is at 0 for all -- the boundary law as a threat
    ranking: an agent self-governs structural guardrails for free but needs the oracle for content.
  - **right -- the cheap safe gate.** The missed-danger rate vs ρ for the load-bearing threats: the
    oracle-in-the-loop drives each to zero at a small consultation budget (the UA9 knee, on safety).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.cu2_threats import CU2ThreatsResult

_KEYED_COLOR = {"structure": "#2ca02c", "near-structure": "#ff7f0e", "content": "#d62728"}


def plot_cu2_threats(result: CU2ThreatsResult, path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax_l = plt.subplots(1, 1, figsize=(8.5, 5.2))

    threats = result.threats
    labels = [f"{t.label}\n({t.keyed})" for t in threats]
    free = [t.free.missed_danger_rate for t in threats]
    colors = [_KEYED_COLOR.get(t.keyed, "#333") for t in threats]
    x = range(len(threats))
    ax_l.bar(x, free, color=colors, alpha=0.9, label="free preview (unverified M_θ)")
    ax_l.plot(list(x), [t.oracle.missed_danger_rate for t in threats], "D", color="#1f77b4",
              ms=10, label="oracle preview (verified)")
    for i, t in enumerate(threats):
        ax_l.text(i, t.free.missed_danger_rate + 0.02, f"{t.free.missed_dangers} run",
                  ha="center", fontsize=8.5, color="#333")
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels(labels, fontsize=9.5)
    ax_l.set_ylabel("missed-danger rate\n(truly-unsafe plans the agent executed)")
    ax_l.set_title("the boundary law as a threat ranking\n"
                   "(an agent self-governs structure for free, needs the oracle for content)")
    ax_l.set_ylim(0, max(free) * 1.3 + 0.05 if free else 1.0)
    ax_l.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "CU2-threats: the safety gate across a cyber threat spectrum "
        "(service kill = structure → credential tampering = content)",
        fontsize=10.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
