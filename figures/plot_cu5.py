"""Plot the CU5 figure (SPEC-22 §5, H97): the closed-loop safe agent.

Two panels from a :class:`~verisim.acd.closed_loop_agent.CU5Result`:

  - **left -- the loop closes.** For the stakes-aware agent, task-success-rate (green, the utility:
    did it finish the job?) rises and the unsafe-episode rate (red, the safety: did it ever do the
    irreversible bad thing?) falls as the consultation budget ρ grows. The free agent (ρ=0) is in
    the **bad corner** -- unsafe *and* unreliable; the oracle agent (ρ=1) is in the **good corner**
    -- safe *and* reliable. The shaded band marks the safe-and-reliable region the agent must reach.
  - **right -- where you spend the budget (the knee).** The unsafe-episode rate vs ρ under a uniform
    schedule (consult a random ρ-fraction) versus a stakes-aware one (consult the actions the model
    is most uncertain about). Uniform buys safety ~linearly; stakes-aware spends the budget on the
    blind spots and reaches zero-harm at a fraction of the budget -- the knee, the design lesson.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.closed_loop_agent import CU5Result


def plot_cu5(result: CU5Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prioritized = [c for c in result.cells if c.schedule == "prioritized"]
    uniform = [c for c in result.cells if c.schedule == "uniform"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    rhos = [c.rho for c in prioritized]
    succ = [c.success_rate for c in prioritized]
    unsafe = [c.unsafe_rate for c in prioritized]
    ax_l.plot(rhos, succ, "-o", color="#2ca02c", lw=2.2, ms=8,
              label="task success (finished the job)")
    ax_l.plot(rhos, unsafe, "-s", color="#d62728", lw=2.2, ms=8,
              label="unsafe (did the irreversible bad thing)")
    ax_l.axhspan(0.95, 1.0, color="#2ca02c", alpha=0.06)
    knee = next((c.rho for c in prioritized if c.unsafe_rate <= 0.05 and c.success_rate >= 0.95),
                None)
    if knee is not None:
        ax_l.axvline(knee, color="#1f77b4", ls=":", lw=1.5,
                     label=f"safe AND reliable at ρ = {knee:g}")
    ax_l.annotate("free agent:\nunsafe + unreliable", xy=(0.0, result.free_unsafe),
                  xytext=(0.12, 0.62), fontsize=8.2,
                  arrowprops={"arrowstyle": "->", "color": "#555"})
    ax_l.set_xlabel("ρ  (the agent's oracle-consultation budget)")
    ax_l.set_ylabel("rate over the episode battery")
    ax_l.set_title(f"the loop closes: grounding makes the agent safe AND reliable (φ={result.phi})",
                   fontsize=9.5)
    ax_l.set_ylim(-0.03, 1.05)
    ax_l.legend(fontsize=8.5, loc="center right")

    ax_r.plot([c.rho for c in uniform], [c.unsafe_rate for c in uniform], "-s",
              color="#9467bd", lw=2, ms=7, label="uniform consultation (random ρ-fraction)")
    ax_r.plot([c.rho for c in prioritized], [c.unsafe_rate for c in prioritized], "-o",
              color="#d62728", lw=2.2, ms=8, label="stakes-aware (consult where uncertain)")
    u_knee = next((c.rho for c in uniform if c.unsafe_rate <= 0.05 and c.success_rate >= 0.95), 1.0)
    p_knee = next((c.rho for c in prioritized if c.unsafe_rate <= 0.05 and c.success_rate >= 0.95),
                  1.0)
    ax_r.set_xlabel("ρ  (the agent's oracle-consultation budget)")
    ax_r.set_ylabel("unsafe-episode rate (the irreversible-harm rate)")
    ax_r.set_title(f"where you spend the budget: stakes-aware buys the knee\n"
                   f"(safe-and-reliable at ρ={p_knee:g} vs uniform's ρ={u_knee:g})", fontsize=9.5)
    ax_r.set_ylim(-0.03, 1.05)
    ax_r.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        "CU5 / H97: the closed-loop safe agent — a verified world model lets an agent finish the "
        "job without the irreversible harm; stakes-aware verification buys it cheaply",
        fontsize=9.4, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
