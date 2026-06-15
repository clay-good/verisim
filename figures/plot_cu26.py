"""Plot the CU26 figure (SPEC-22 H119): the low-and-slow danger -- target the accumulator.

Two panels from a :class:`~verisim.acd.cumulative_targeting.CU26Result` -- the framework applied to
a *cumulative* danger no single action realizes (mass collection / data hoarding):

  - **left -- the worst-case (low-and-slow) cost/safety frontier.** Adversarial breach vs oracle
    calls. The uniform blind clock is flat at 1.0 until the full oracle (the knee is a mirage). The
    real-world magnitude / DLP heuristic is the red X stuck at breach 1.0 (cheap, but blind to a
    spread-out collection -- ``covers`` predicted this a priori). The grammar target (watch every
    sensitive flow) is the orange diamond: safe but overpaying. Only the framework-*derived*
    accumulator-closure is the green star in the safe-and-cheapest corner.
  - **right -- the cost law.** Mean oracle calls vs the budget ``B``. A higher threshold makes the
    closure boundary *rarer* (cheaper) while the grammar surface is unchanged, so the closure's cost
    advantage grows with ``B``. The magnitude curve is cheap but dashed -- it leaks at every ``B``.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.cumulative_targeting import CU26Result


def plot_cu26(result: CU26Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.2, 4.9))
    full = result.full_oracle
    by = {c.name: c for c in result.candidates}
    magnitude, grammar, closure = by["magnitude"], by["grammar"], by["closure"]

    # left: worst-case (low-and-slow) cost/safety frontier
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.adversarial_breach for c in result.uniform]
    ax_l.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind clock)")
    ax_l.plot(magnitude.mean_calls, magnitude.adversarial_breach, "X", color="#d62728", ms=14,
              label="magnitude / DLP heuristic: leaks", zorder=5)
    ax_l.plot(grammar.mean_calls, grammar.adversarial_breach, "D", color="#ff7f0e", ms=11,
              label="grammar (watch every sensitive flow): safe, overpays", zorder=5)
    ax_l.plot(closure.mean_calls, closure.adversarial_breach, "*", color="#2ca02c", ms=22,
              label="accumulator-closure (DERIVED): safe + cheapest", zorder=6)
    ax_l.annotate(
        "magnitude / DLP heuristic:\ncovers()=False → leaks to low-and-slow\n(predicted a priori)",
        xy=(magnitude.mean_calls, magnitude.adversarial_breach),
        xytext=(magnitude.mean_calls + 7, 0.80), fontsize=8.0, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )
    ax_l.annotate(
        f"covers()=True → 0 breach\n{closure.mean_calls:.2f} calls\n"
        f"({grammar.mean_calls / closure.mean_calls:.0f}x cheaper than grammar)",
        xy=(closure.mean_calls, closure.adversarial_breach),
        xytext=(closure.mean_calls + 9, 0.18), fontsize=8.2, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )
    ax_l.annotate(
        f"full oracle\n{full.mean_calls:.0f} calls",
        xy=(full.mean_calls, full.adversarial_breach), xytext=(full.mean_calls - 16, 0.16),
        fontsize=8.2, color="#555", arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_l.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_l.set_ylabel("low-and-slow breach rate  (the worst-case safety axis)")
    ax_l.set_ylim(-0.05, 1.10)
    ax_l.set_title("a cumulative danger no single action realizes (mass collection / hoarding)",
                   fontsize=9.0)
    ax_l.legend(fontsize=7.6, loc="center right")

    # right: the cost law -- mean calls vs the budget B
    sweep = result.cost_by_budget
    bs = sorted(sweep)
    if bs:
        mag = [sweep[b]["magnitude"] for b in bs]
        gram = [sweep[b]["grammar"] for b in bs]
        clo = [sweep[b]["closure"] for b in bs]
        ax_r.plot(bs, gram, "-D", color="#ff7f0e", lw=2.0, ms=9, label="grammar (covers, overpays)")
        ax_r.plot(bs, mag, "--X", color="#d62728", lw=1.8, ms=11,
                  label="magnitude (cheap, LEAKS at every B)")
        ax_r.plot(bs, clo, "-*", color="#2ca02c", lw=2.0, ms=18, label="closure (covers, cheapest)")
        for b, cv in zip(bs, clo, strict=True):
            ratio = sweep[b]["grammar"] / cv if cv > 0 else float("inf")
            ax_r.annotate(f"{ratio:.0f}x", xy=(b, cv), xytext=(b, cv + 0.6), ha="center",
                          fontsize=8.0, color="#2ca02c", fontweight="bold")
        ax_r.set_xticks(bs)
    ax_r.set_xlabel("budget  B  (concurrent sensitive flows that count as mass collection)")
    ax_r.set_ylabel("mean oracle calls / deployment")
    ax_r.set_title("the cost law: a higher budget makes the closure boundary rarer (cheaper);\n"
                   "the grammar surface is unchanged → closure's advantage grows with B",
                   fontsize=9.0)
    ax_r.legend(fontsize=8.0, loc="center right")

    fig.suptitle(
        "CU26 / H119: the framework is generative for a CUMULATIVE danger — a low-and-slow "
        "collection no single action realizes.\nCumulative coverage predicts the real-world "
        "magnitude/DLP heuristic leaks and derives the accumulator-closure (safe + cheapest)  "
        f"({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=8.7, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
