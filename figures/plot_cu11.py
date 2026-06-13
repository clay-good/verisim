"""Plot the CU11 figure (SPEC-22 H104): un-gameable targeting -- the adversary controls the timing.

Two panels from a :class:`~verisim.acd.adversarial_targeting.CU11Result`:

  - **left -- random vs adversarial breach, by schedule.** Paired bars at the uniform knee budget,
    model self-targeting, and structure targeting: the random-timing breach (what CU10 measured)
    next to the worst-case-over-timing adversarial breach. Uniform and model jump toward 1.0 when an
    attacker who picks the step; structure stays flat at the oracle's 0.
  - **right -- the uniform knee is a mirage.** Uniform breach vs ρ: the random-timing curve (CU9's
    knee, falling) against the adversarial curve (flat near 1.0 until the full oracle), with
    structure's un-gameable 0 line for reference. Average-case safety the schedule bought on a
    random workload evaporates against an adversary who controls *when* danger happens.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.adversarial_targeting import CU11Result


def plot_cu11(result: CU11Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    # left: random vs adversarial breach by schedule (uniform knee / model / structure)
    knee = min(
        (c for c in result.uniform if c.rho is not None and 0.0 < c.rho < 1.0),
        key=lambda c: c.random_breach, default=result.uniform[0],
    )
    groups = [
        (f"uniform\nρ={knee.rho:g}", knee.random_breach, knee.adversarial_breach, knee.mean_calls),
        ("model\nself-target", result.model.random_breach, result.model.adversarial_breach,
         result.model.mean_calls),
        ("structure\ncrown-jewel", result.structure.random_breach,
         result.structure.adversarial_breach, result.structure.mean_calls),
    ]
    xs = range(len(groups))
    w = 0.38
    ax_l.bar([x - w / 2 for x in xs], [g[1] for g in groups], width=w, color="#7f7fbf",
             label="random timing (CU10)")
    ax_l.bar([x + w / 2 for x in xs], [g[2] for g in groups], width=w, color="#d62728",
             label="adversarial timing (CU11)")
    for x, g in zip(xs, groups, strict=True):
        ax_l.text(x + w / 2, g[2] + 0.02, f"{g[2]:.2f}", ha="center", va="bottom", fontsize=8.0)
        ax_l.text(x - w / 2, g[1] + 0.02, f"{g[1]:.2f}", ha="center", va="bottom", fontsize=8.0,
                  color="#444")
        ax_l.text(x, -0.085, f"{g[3]:.1f} calls", ha="center", va="top", fontsize=7.6,
                  color="#2ca02c")
    ax_l.axhline(0.0, color="#2ca02c", lw=1.0, ls=":")
    ax_l.set_xticks(list(xs))
    ax_l.set_xticklabels([g[0] for g in groups], fontsize=8.4)
    ax_l.set_ylabel("breach rate over the deployment")
    ax_l.set_ylim(-0.13, 1.1)
    ax_l.set_title("structure targeting is the only schedule the attacker can't time around",
                   fontsize=9.0)
    ax_l.legend(fontsize=8.0, loc="upper right")

    # right: the uniform knee is a mirage -- random vs adversarial breach vs ρ
    ux = [c.rho for c in result.uniform]
    rand = [c.random_breach for c in result.uniform]
    adv = [c.adversarial_breach for c in result.uniform]
    ax_r.plot(ux, rand, "-o", color="#7f7fbf", lw=2.0, ms=6,
              label="uniform, random timing (CU9 knee)")
    ax_r.plot(ux, adv, "-X", color="#d62728", lw=2.0, ms=8,
              label="uniform, adversarial timing")
    ax_r.axhline(result.structure.adversarial_breach, color="#2ca02c", lw=2.0, ls="--",
                 label="structure, adversarial (un-gameable)")
    ax_r.fill_between(ux, rand, adv, color="#d62728", alpha=0.10)
    ax_r.set_xlabel("verification budget ρ  (uniform schedule)")
    ax_r.set_ylabel("breach rate over the deployment")
    ax_r.set_ylim(-0.03, 1.05)
    ax_r.set_title("the uniform knee is a mirage: adversarial timing erases it", fontsize=9.0)
    ax_r.legend(fontsize=8.0, loc="center left")

    fig.suptitle(
        "CU11 / H104: un-gameable targeting — uniform/model key on signals the attacker controls "
        "(clock phase, the omitting model);\nstructure keys on a grammar-fixed danger surface it "
        f"cannot move  ({result.n_episodes} deployments, real M_θ)",
        fontsize=9.2, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
