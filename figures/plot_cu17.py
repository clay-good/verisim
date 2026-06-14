"""Plot the CU17 figure (SPEC-22 H110): the genesis-grammar boundary.

Two panels from a :class:`~verisim.acd.segmentation_targeting.CU17Result`:

  - **left -- the cost/safety frontier (random timing).** Breach rate vs oracle calls. Uniform is
    the blind curve that only reaches zero breach at the full oracle (top-right). The CU10-CU16
    ``connect`` target is the red X stuck at the free breach rate (cheap, but blind to the config
    genesis -- it does not transfer). The syntactic ``grammar`` target is the orange diamond:
    near-zero breach but well to the right (it overpays, verifying every ``link_up``). Only the
    semantic ``closure`` target is the green star in the safe-and-cheap corner.
  - **right -- the gameability axis (random vs adversarial breach).** Paired bars per targeted
    schedule with calls annotated: ``connect`` breaches at the free rate either way; the syntactic
    ``grammar`` target is near-safe on random workloads but **gameable** (an adversary exposes a
    jewel through a multi-hop intermediate it cannot name); only ``closure`` is un-gameable (0/0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.segmentation_targeting import CU17Result


def plot_cu17(result: CU17Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))
    full = result.uniform[-1]
    connect, grammar, closure = result.connect, result.grammar, result.closure

    # left: cost/safety frontier (random timing)
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.random_breach for c in result.uniform]
    ax_l.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind clock)")
    ax_l.plot(connect.mean_calls, connect.random_breach, "X", color="#d62728", ms=14,
              label="connect (CU10 target): blind", zorder=5)
    ax_l.plot(grammar.mean_calls, grammar.random_breach, "D", color="#ff7f0e", ms=11,
              label="grammar (syntactic): overpays", zorder=5)
    ax_l.plot(closure.mean_calls, closure.random_breach, "*", color="#2ca02c", ms=22,
              label="closure (reachability): safe + cheap", zorder=6)
    ax_l.annotate(
        f"connect target does not transfer\n(breach {connect.random_breach:.2f}, "
        f"the free rate, at {connect.mean_calls:.1f} calls)",
        xy=(connect.mean_calls, connect.random_breach),
        xytext=(connect.mean_calls + 6, 0.86), fontsize=8.0, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )
    ax_l.annotate(
        f"closure: 0 breach\n{closure.mean_calls:.1f} calls "
        f"({full.mean_calls / closure.mean_calls:.0f}x cheaper)",
        xy=(closure.mean_calls, closure.random_breach),
        xytext=(closure.mean_calls + 9, 0.20), fontsize=8.2, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )
    ax_l.annotate(
        f"full oracle\n{full.mean_calls:.0f} calls",
        xy=(full.mean_calls, full.random_breach), xytext=(full.mean_calls - 15, 0.16),
        fontsize=8.2, color="#555", arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_l.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_l.set_ylabel("breach rate  (the safety axis)")
    ax_l.set_ylim(-0.05, 1.08)
    ax_l.set_title("the cheap connect-target does not transfer to a richer danger genesis",
                   fontsize=9.0)
    ax_l.legend(fontsize=7.8, loc="center right")

    # right: random vs adversarial breach by targeted schedule
    schedules = [connect, grammar, closure]
    labels = ["connect\n(CU10 target)", "grammar\n(syntactic)", "closure\n(reachability)"]
    x = np.arange(len(schedules))
    w = 0.38
    rand = [c.random_breach for c in schedules]
    adv = [c.adversarial_breach for c in schedules]
    ax_r.bar(x - w / 2, rand, w, color="#7f7fbf", label="random timing")
    ax_r.bar(x + w / 2, adv, w, color="#d62728", label="adversarial (worst case)")
    for xi, c in zip(x, schedules, strict=True):
        top = max(c.random_breach, c.adversarial_breach)
        ax_r.annotate(f"{c.mean_calls:.1f} calls", xy=(xi, top + 0.03), ha="center", va="bottom",
                      fontsize=7.8, color="#333")
    ax_r.axhline(0.0, color="#999", lw=0.8)
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(labels, fontsize=8.4)
    ax_r.set_ylabel("breach rate over the deployment")
    ax_r.set_ylim(-0.03, 1.16)
    ax_r.set_title("syntactic targeting is gameable; the reachability closure is not", fontsize=9.0)
    ax_r.legend(fontsize=8.0, loc="upper center")

    fig.suptitle(
        "CU17 / H110: the targeting result needs the danger's genesis grammar — a jewel's "
        "exposure is born by the\nconfig grammar (link/svc/host), not by connect, so you target "
        "the reachability closure (semantic), not an action class (syntactic)  "
        f"({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=8.8, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
