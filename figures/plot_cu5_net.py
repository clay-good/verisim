"""Plot the CU5-net figure (SPEC-22 H100): the closed loop on a REAL trained network model.

Two panels from a :class:`~verisim.acd.closed_loop_net.CU5NetResult`:

  - **left -- the loop closes on the real model.** Task success (green, did the agent establish the
    work connectivity?) and the unsafe rate (red, did it ever open an exfiltration flow to a
    protected host?) vs the consultation budget ρ. The free agent (ρ=0) is in the bad corner --
    unsafe (the model's flow drift hides real exfil) and, where the model hallucinates flows,
    unreliable; the oracle agent (ρ=1) is in the good corner -- safe and reliable. This is a real
    transformer world-model, not a φ-dial stand-in.
  - **right -- safety bought by consultation.** The mean number of missed exfiltration flows per
    episode and the mean oracle calls vs ρ: the budget the agent spends re-anchoring its belief to
    the truth, and the exfiltration it buys down. The knee is where a few calls remove the danger.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.closed_loop_net import CU5NetResult


def plot_cu5_net(result: CU5NetResult, path: str | Path) -> Path:  # pragma: no cover - local plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = [c.rho for c in result.cells]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    ax_l.plot(rhos, [c.success_rate for c in result.cells], "-o", color="#2ca02c", lw=2.3, ms=8,
              label="task success (established work connectivity)")
    ax_l.plot(rhos, [c.unsafe_rate for c in result.cells], "-s", color="#d62728", lw=2.3, ms=8,
              label="unsafe (opened an exfiltration flow)")
    ax_l.axhspan(0.95, 1.0, color="#2ca02c", alpha=0.06)
    ax_l.annotate("free agent: opens EVERY exfil flow", xy=(0.0, result.free_unsafe),
                  xytext=(0.08, 0.40), fontsize=8.2,
                  arrowprops={"arrowstyle": "->", "color": "#d62728"})
    ax_l.annotate("success flat at 1.0 — the real drift is one-sided\n"
                  "(the model OMITS exfil, never hallucinates it)", xy=(0.5, 1.0),
                  xytext=(0.16, 0.83), fontsize=7.8, color="#2ca02c")
    ax_l.set_xlabel("ρ  (the agent's oracle-consultation budget)")
    ax_l.set_ylabel("rate over the contested episodes")
    ax_l.set_title(f"verification closes the SAFETY axis on a real trained M_θ  "
                   f"({result.n_episodes} eps)", fontsize=9.2)
    ax_l.set_ylim(-0.03, 1.05)
    ax_l.legend(fontsize=8.4, loc="center right")

    ax_r.plot(rhos, [c.mean_unsafe for c in result.cells], "-s", color="#d62728", lw=2.2, ms=7,
              label="mean missed exfil flows / episode")
    ax_r.set_xlabel("ρ  (the agent's oracle-consultation budget)")
    ax_r.set_ylabel("mean missed exfiltration flows", color="#d62728")
    ax_r.tick_params(axis="y", labelcolor="#d62728")
    ax_r.set_title("safety bought by consultation (the network defender)", fontsize=9.5)
    ax2 = ax_r.twinx()
    ax2.plot(rhos, [c.mean_calls for c in result.cells], "-^", color="#1f77b4", lw=1.8, ms=6,
             label="mean oracle calls / episode")
    ax2.set_ylabel("mean oracle calls / episode", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    lines = ax_r.get_lines() + ax2.get_lines()
    ax_r.legend(lines, [ln.get_label() for ln in lines], fontsize=8.2, loc="center right")

    fig.suptitle(
        "CU5-net / H100: the closed loop's SAFETY axis closes on a REAL learned model — free agents"
        " open every exfil flow, verification removes them; the real drift is one-sided",
        fontsize=9.2, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
