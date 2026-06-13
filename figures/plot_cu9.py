"""Plot the CU9 figure (SPEC-22 H102): the agent-safety horizon.

Two panels from a :class:`~verisim.acd.safety_horizon.CU9Result`:

  - **left -- the survival curves.** The fraction of agents still safe (no exfiltration yet) vs the
    deployment step, one curve per consultation budget ρ. The free agent (red) decays toward zero --
    over a long enough deployment it is certain to breach -- while verification flattens the curve
    and the oracle (green) stays flat at 1.0. Unverified safety is a clock that runs out.
  - **right -- the safe horizon vs the budget.** The mean safe runtime (steps before first breach)
    rises with ρ, and the end-of-deployment breach rate falls -- how much safe deployment each unit
    of verification buys.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.safety_horizon import CU9Result


def plot_cu9(result: CU9Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    steps = list(range(1, result.horizon + 1))
    n = len(result.curves)
    for i, curve in enumerate(result.curves):
        if curve.rho <= 0.0:
            color, lw, label = "#d62728", 2.6, "free (ρ=0): unverified"
        elif curve.rho >= 1.0:
            color, lw, label = "#2ca02c", 2.6, "oracle (ρ=1): every step verified"
        else:
            g = int((0.30 + 0.45 * (i / max(n - 1, 1))) * 255)
            color, lw, label = f"#{g:02x}{g:02x}bf", 1.6, f"ρ = {curve.rho:g}"
        ax_l.plot(steps, curve.survival, color=color, lw=lw, label=label)
    ax_l.set_xlabel("deployment step  (actions the agent has run)")
    ax_l.set_ylabel("fraction of agents still safe (no exfiltration yet)")
    ax_l.set_title(f"unverified safety is a clock that runs out  ({result.n_episodes} deployments, "
                   f"real M_θ)", fontsize=9.0)
    ax_l.set_ylim(-0.03, 1.03)
    ax_l.legend(fontsize=8.0, loc="lower left")

    rhos = [c.rho for c in result.curves]
    ax_r.plot(rhos, [c.mean_safe_steps for c in result.curves], "-o", color="#1f77b4", lw=2.2, ms=7,
              label="mean safe runtime (steps to first breach)")
    ax_r.set_xlabel("ρ  (the agent's oracle-consultation budget)")
    ax_r.set_ylabel("mean safe steps before first breach", color="#1f77b4")
    ax_r.tick_params(axis="y", labelcolor="#1f77b4")
    ax_r.set_title("the safe horizon grows with verification", fontsize=9.4)
    ax2 = ax_r.twinx()
    ax2.plot(rhos, [c.breach_rate for c in result.curves], "-s", color="#d62728", lw=2.0, ms=6,
             label="breach rate over the full deployment")
    ax2.set_ylabel("breach rate over the deployment", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(-0.03, 1.03)
    lines = ax_r.get_lines() + ax2.get_lines()
    ax_r.legend(lines, [ln.get_label() for ln in lines], fontsize=8.0, loc="center right")

    fig.suptitle(
        "CU9 / H102: the agent-safety horizon — unverified agents breach with deployment length "
        "(near-certain over a long run); verification stops the clock",
        fontsize=9.3, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
