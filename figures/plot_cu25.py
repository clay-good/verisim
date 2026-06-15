"""Plot the CU25 figure (SPEC-22 H118): the composite under real drift -- foresight is not safety.

Two panels from a :class:`~verisim.acd.composite_trained.CU25Result` (the trained-arm closure of the
CU24 composite, run on the real network ``M_theta``):

  - **left -- the boundary law at the composite.** Per leg, the model's self-governance recall (how
    much of the leg's danger its own preview foresees) beside the model-self-targeting adversarial
    breach. The model sees the direct-structural outage leg well (~0.79) and is blind on the content
    exfil (~0.08) and the multi-hop exposure (~0.02) -- the boundary splits the threat model. The
    punchline is the outage pair: 0.79 recall yet ~1.000 breach -- high average foresight buys no
    worst-case safety, because the adversary needs one blind spot.
  - **right -- the cost/safety frontier.** Model self-targeting is a failed red cluster (breach
    ~1.0); only the model-free union target is the safe-and-cheap green star (0 breach at the union
    surface cost), model-independently, far below the full oracle (verify every step).
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.composite_targeting import LEG_NAMES
from verisim.acd.composite_trained import CU25Result

_LEG_DISPLAY = {
    "exfil": "exfil\n(content)",
    "exposure": "exposure\n(multi-hop)",
    "outage": "outage\n(structural)",
}


def plot_cu25(result: CU25Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))
    legs = list(LEG_NAMES)
    xs = list(range(len(legs)))
    recall = [result.self_gov_recall[leg] for leg in legs]
    breach = [result.model_adv_breach[leg] for leg in legs]

    # left: per-leg self-governance recall vs model-self-targeting adversarial breach.
    w = 0.38
    ax_l.bar([x - w / 2 for x in xs], recall, width=w, color="#1f77b4",
             label="self-governance recall (model foresees the danger)")
    ax_l.bar([x + w / 2 for x in xs], breach, width=w, color="#d62728",
             label="model-self-targeting adversarial breach")
    for x, r, b in zip(xs, recall, breach, strict=True):
        ax_l.text(x - w / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8.6, color="#1f77b4")
        ax_l.text(x + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8.6, color="#d62728")
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels([_LEG_DISPLAY[leg] for leg in legs], fontsize=8.6)
    ax_l.set_ylabel("rate")
    ax_l.set_ylim(0, 1.16)
    ax_l.annotate(
        "0.79 foreseen,\nstill 1.00 breached", xy=(2 + w / 2, breach[2]),
        xytext=(1.05, 1.05), fontsize=8.2, color="#8b0000",
        arrowprops=dict(arrowstyle="->", color="#8b0000", lw=1.1),
    )
    ax_l.set_title(
        "the boundary law at the composite: blind on the content leg, partial on the\n"
        "structural legs — yet even 0.78 foresight is adversarially breached 1.00",
        fontsize=8.4,
    )
    ax_l.legend(fontsize=7.4, loc="upper left")

    # right: cost/safety frontier on the real model.
    saving = (
        result.full_oracle_calls / result.union_target_calls
        if result.union_target_calls else float("inf")
    )
    ax_r.plot(result.model_calls, result.model_composite_adv, "X", color="#d62728", ms=14,
              label="model self-targeting (fails the composite)", zorder=5)
    ax_r.plot(result.union_target_calls, 0.0, "*", color="#2ca02c", ms=22,
              label="model-free union target (every leg)", zorder=6)
    ax_r.plot(result.full_oracle_calls, 0.0, "s", color="#555555", ms=10,
              label="full oracle (verify every step)")
    ax_r.annotate(
        f"union target: 0 breach\n{result.union_target_calls:.1f} calls ({saving:.0f}x cheaper)",
        xy=(result.union_target_calls, 0.0),
        xytext=(result.union_target_calls + 6, 0.22), fontsize=8.4, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )
    ax_r.annotate(
        f"model self-targeting\nbreach {result.model_composite_adv:.2f}",
        xy=(result.model_calls, result.model_composite_adv),
        xytext=(result.model_calls + 8, 0.80), fontsize=8.4, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
    )
    ax_r.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_r.set_ylabel("composite adversarial breach  (the safety axis)")
    ax_r.set_xlim(-2, result.full_oracle_calls + 6)
    ax_r.set_ylim(-0.06, 1.10)
    ax_r.set_title("only the model-free union target is safe on the whole threat model",
                   fontsize=9.0)
    ax_r.legend(fontsize=7.6, loc="center right")

    fig.suptitle(
        "CU25 / H118: the composite under real drift — the trained net M_θ is blind on the content "
        "exfil leg (recall 0.07) and\npartial on the structural legs (0.57/0.78), but EVERY leg is "
        "adversarially breached 1.00; only the model-free union target is safe on "
        f"every leg  ({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=8.4, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:  # pragma: no cover - local plotting
    import argparse

    from verisim.acd.composite_trained import run_cu25
    from verisim.experiments.flagship import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/net-l")
    parser.add_argument("--out", type=str, default="figures/cu25_composite_trained.png")
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint).world_model
    plot_cu25(run_cu25(model), args.out)


if __name__ == "__main__":
    main()
