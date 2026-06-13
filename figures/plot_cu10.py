"""Plot the CU10 figure (SPEC-22 H103): targeted verification -- what to verify beats how much.

Two panels from a :class:`~verisim.acd.targeted_verification.CU10Result`:

  - **left -- the cost/safety frontier.** Breach rate vs mean oracle calls. The *uniform* (blind)
    schedule traces a slow curve that only reaches zero breach at the full oracle (top right of its
    spend). *Model self-targeting* sits at the free agent's breach rate despite spending calls -- it
    cannot see its own omissions (CU8). *Structure targeting* is a single dominating point: the
    oracle's zero breach at a small fraction of the calls.
  - **right -- breach rate by strategy, with the cost annotated.** Free / model / structure / oracle
    bars; model self-targeting ≈ free (fails), structure ≈ oracle (succeeds) but far cheaper.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.targeted_verification import CU10Result


def plot_cu10(result: CU10Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    # left: the cost/safety frontier
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.breach_rate for c in result.uniform]
    ax_l.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind, CU9): sweep ρ")
    ax_l.plot([result.model.mean_calls], [result.model.breach_rate], "X", color="#d62728", ms=15,
              label=f"model self-targeting ({result.model.mean_calls:.1f} calls): fails")
    ax_l.plot([result.structure.mean_calls], [result.structure.breach_rate], "*", color="#2ca02c",
              ms=20, label=f"structure / crown-jewel ({result.structure.mean_calls:.1f} calls): "
              f"oracle-safe")
    ax_l.annotate(
        "oracle safety,\nfor ~%.0fx fewer calls" % (
            result.uniform[-1].mean_calls / max(result.structure.mean_calls, 1e-9)),
        xy=(result.structure.mean_calls, result.structure.breach_rate),
        xytext=(result.structure.mean_calls + 0.18 * max(ux), 0.18),
        fontsize=8.2, color="#2ca02c",
        arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
    )
    ax_l.set_xlabel("mean oracle calls per deployment  (the verification cost)")
    ax_l.set_ylabel("breach rate over the deployment  (the danger)")
    ax_l.set_title(f"what to verify beats how much  ({result.n_episodes} deployments, real M_θ)",
                   fontsize=9.2)
    ax_l.set_ylim(-0.03, 1.03)
    ax_l.legend(fontsize=8.0, loc="upper right")

    # right: breach rate by strategy, cost annotated
    free = result.uniform[0]
    full = result.uniform[-1]
    bars = [
        ("free\n(ρ=0)", free.breach_rate, free.mean_calls, "#d62728"),
        ("model\nself-target", result.model.breach_rate, result.model.mean_calls, "#d62728"),
        ("structure\ncrown-jewel", result.structure.breach_rate, result.structure.mean_calls,
         "#2ca02c"),
        ("oracle\n(ρ=1)", full.breach_rate, full.mean_calls, "#2ca02c"),
    ]
    xs = range(len(bars))
    ax_r.bar(xs, [b[1] for b in bars], color=[b[3] for b in bars], width=0.62)
    for x, (_, br, calls, _color) in zip(xs, bars, strict=True):
        ax_r.text(x, br + 0.03, f"{calls:.1f}\ncalls", ha="center", va="bottom", fontsize=8.0)
    ax_r.set_xticks(list(xs))
    ax_r.set_xticklabels([b[0] for b in bars], fontsize=8.4)
    ax_r.set_ylabel("breach rate over the deployment")
    ax_r.set_ylim(0, 1.15)
    ax_r.set_title("the model can't self-target its omissions; structure can", fontsize=9.2)

    fig.suptitle(
        "CU10 / H103: targeted verification — you can't ask the omitting model where it's wrong "
        "(CU8); the defender's crown-jewel knowledge can, and danger is cheap to defend",
        fontsize=9.0, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
