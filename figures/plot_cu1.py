"""Plot the CU1 figure (SPEC-22 §4, H93): the agent-in-the-loop safety gate.

Two panels from a :class:`~verisim.experiments.cu_safety_gate.CU1Result`:

  - **left -- missed danger by preview and guardrail.** The fraction of truly-unsafe plans the agent
    *executed* because its preview said "safe". The free (unverified `M_θ`) preview misses real
    dangers on the **content** guardrail (``/passwd`` overwrite: the agent runs credential-
    corrupting plans), while the oracle misses none; on the **structure** guardrail (a protected
    that survives -- dynamics the model learns faithfully) the free preview already gates correctly.
  - **right -- the cheap safe gate.** The missed-danger rate on the content guardrail vs the
    ρ-consultation budget: re-anchoring the preview to the oracle drives it to zero at a small ρ --
    safe gating bought back sub-linearly (the SPEC-19/UA9 knee, here on *agent safety*).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.cu_safety_gate import CU1Result, knee_rho


def plot_cu1(result: CU1Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))

    # left: missed-danger by (guardrail, preview)
    groups = [result.content, result.structure]
    labels = [f"content\n({g.guardrail})" if g is result.content else "structure\n(proc alive)"
              for g in groups]
    free = [g.free.missed_danger_rate for g in groups]
    oracle = [g.oracle.missed_danger_rate for g in groups]
    x = range(len(groups))
    w = 0.36
    ax_l.bar([i - w / 2 for i in x], free, w, color="#d62728", label="free preview (unverified)")
    ax_l.bar([i + w / 2 for i in x], oracle, w, color="#1f77b4", label="oracle preview (verified)")
    for i, g in enumerate(groups):
        ax_l.text(i - w / 2, g.free.missed_danger_rate + 0.02,
                  f"{g.free.missed_dangers} run", ha="center", fontsize=8, color="#d62728")
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels(labels, fontsize=9)
    ax_l.set_ylabel("missed-danger rate\n(truly-unsafe plans the agent executed)")
    ax_l.set_title("an unverified preview executes destructive plans")
    ax_l.set_ylim(0, 1.05)
    ax_l.legend(fontsize=8, loc="upper right")

    # right: the missed-danger knee vs ρ on the content guardrail
    knee = sorted(result.content.knee, key=lambda x: x[0])
    rhos = [r for r, _ in knee]
    mdr = [o.missed_danger_rate for _, o in knee]
    ax_r.plot(rhos, mdr, "-o", color="#d62728")
    safe_rho = knee_rho(result.content)
    ax_r.axvline(safe_rho, color="#2ca02c", ls=":", label=f"missed-danger = 0 at ρ = {safe_rho:g}")
    ax_r.axhline(0, color="#888", lw=0.8)
    ax_r.set_xlabel("ρ  (oracle-consultation budget of the preview)")
    ax_r.set_ylabel("missed-danger rate (content guardrail)")
    ax_r.set_title("verification buys a safe gate cheaply (the knee)")
    ax_r.set_ylim(-0.03, max(mdr) * 1.1 + 0.03 if mdr else 1.0)
    ax_r.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "CU1 / H93: a computer-use agent needs a verified world model to gate its actions safely — "
        "and the oracle buys that safety cheaply"
    )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
