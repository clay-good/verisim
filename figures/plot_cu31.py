"""Plot the CU31 figure (SPEC-22 H124): the concurrent (multi-agent) safety gate.

Two panels from a :class:`~verisim.acd.concurrent_targeting.CU31Result` -- the targeting result
carried from one agent to a FLEET sharing one network:

  - **left -- the worst-case cost/safety frontier for the joint danger.** Adversarial (fleet
    low-and-slow) breach vs oracle calls. The uniform blind clock is flat at 1.0 (the knee is a
    mirage). The per-agent gate (the CU26 closure scoped to one principal) is the red X stuck at
    breach 1.0 -- cheap, but it watches the wrong scope, so a fleet adversary spreads the collection
    under it (``covers``=False, predicted a priori). The shared grammar (watch every sensitive flow)
    is the orange diamond: safe but overpaying. Only the shared closure on the JOINT accumulator is
    the green star in the safe-and-cheapest corner.
  - **right -- the fragmentation law.** Adversarial breach vs fleet size ``K``. The per-agent gate
    covers iff a single principal must hold ``>= B`` (``K=1``) and leaks the instant the fleet
    fragments (``K>=2``); the shared gate is flat at 0 -- invariant to fragmentation.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.concurrent_targeting import CU31Result


def plot_cu31(
    result: CU31Result, verdict: dict[str, object], path: str | Path
) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.4, 5.0))
    full = result.full_oracle
    by = {c.name: c for c in result.candidates}
    per_agent, grammar, closure = by["per_agent"], by["shared_grammar"], by["shared_closure"]

    # left: worst-case cost/safety frontier for the joint danger
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.adversarial_breach for c in result.uniform]
    ax_l.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind clock): mirage")
    ax_l.plot(per_agent.mean_calls, per_agent.adversarial_breach, "X", color="#d62728", ms=15,
              label="per-agent budget (CU26 scoped per principal): LEAKS", zorder=5)
    ax_l.plot(grammar.mean_calls, grammar.adversarial_breach, "D", color="#ff7f0e", ms=11,
              label="shared grammar (watch every sensitive flow): safe, overpays", zorder=5)
    ax_l.plot(closure.mean_calls, closure.adversarial_breach, "*", color="#2ca02c", ms=23,
              label="shared closure (DERIVED, JOINT accumulator): safe + cheapest", zorder=6)
    ax_l.annotate(
        "per-agent gate:\ncovers(joint)=False → leaks to a\nfleet that spreads the collection\n"
        "(the right rule, the wrong scope)",
        xy=(per_agent.mean_calls, per_agent.adversarial_breach),
        xytext=(per_agent.mean_calls + 6, 0.74), fontsize=8.0, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )
    ratio_g = grammar.mean_calls / closure.mean_calls if closure.mean_calls > 0 else float("inf")
    ax_l.annotate(
        f"covers(joint)=True → 0 breach\n{closure.mean_calls:.2f} calls\n"
        f"({ratio_g:.0f}x cheaper than grammar)",
        xy=(closure.mean_calls, closure.adversarial_breach),
        xytext=(closure.mean_calls + 9, 0.20), fontsize=8.2, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )
    ax_l.annotate(
        f"full oracle\n{full.mean_calls:.0f} calls",
        xy=(full.mean_calls, full.adversarial_breach), xytext=(full.mean_calls - 16, 0.18),
        fontsize=8.2, color="#555", arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_l.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_l.set_ylabel("fleet low-and-slow breach rate  (the worst-case safety axis)")
    ax_l.set_ylim(-0.05, 1.10)
    ax_l.set_title(
        f"JOINT mass-collection danger (B={result.budget}) across a fleet (K={result.n_agents})",
        fontsize=9.0)
    ax_l.legend(fontsize=7.4, loc="center right")

    # right: the fragmentation law -- adversarial breach vs fleet size K
    ks = sorted(result.by_n_agents)
    if ks:
        pa = [result.by_n_agents[k]["per_agent"][0] for k in ks]
        sh = [result.by_n_agents[k]["shared_closure"][0] for k in ks]
        ax_r.plot(ks, pa, "-X", color="#d62728", lw=2.2, ms=13,
                  label="per-agent budget (leaks once K>=2)")
        ax_r.plot(ks, sh, "-*", color="#2ca02c", lw=2.2, ms=18,
                  label="shared closure (invariant to fragmentation)")
        for k in ks:
            cov = result.by_n_agents[k]["per_agent"][2]
            ax_r.annotate("covers" if cov else "LEAKS",
                          xy=(k, result.by_n_agents[k]["per_agent"][0]),
                          xytext=(k, result.by_n_agents[k]["per_agent"][0] + 0.06), ha="center",
                          fontsize=8.0, color="#2ca02c" if cov else "#d62728", fontweight="bold")
        ax_r.axvspan(0.5, 1.5, color="#2ca02c", alpha=0.06)
        ax_r.set_xticks(ks)
    ax_r.set_ylim(-0.08, 1.12)
    ax_r.set_xlabel("fleet size  K  (number of agents sharing the network)")
    ax_r.set_ylabel("fleet low-and-slow breach rate")
    ax_r.set_title("the fragmentation law: the per-agent gate covers iff a single principal must\n"
                   "hold >= B (K=1); it leaks the instant the fleet fragments. Shared gate does"
                   " not.",
                   fontsize=9.0)
    ax_r.legend(fontsize=8.0, loc="center right")

    fig.suptitle(
        "CU31 / H124: you cannot defend a shared resource with per-agent budgets — the "
        "multi-principal coverage law.\nA per-agent gate (covering per principal) LEAKS the JOINT "
        "danger across a fleet; only a shared gate over the merged action stream covers it, "
        f"un-gameably and cheaply  ({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=8.7, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
