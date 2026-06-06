"""Plot the ED6 two-oracle learned figure (SPEC-7 §10.1, DS8): H12 on the **real** `M_θ`.

The learned-arm companion to :mod:`figures.plot_ed6_two_oracle` (the synthetic, mode-split version).
One trained flat `M_θ` has a single error distribution (no ``gross``/``subtle`` dial), so this is
one panel pair from one model:

  - **left** -- the three H12 rates with bootstrap CIs: **non-redundant** (≈ 0 by construction --
    the cheap consistency oracle catches nothing the bit-exact oracle misses, so it is *redundant
    for verification*), **consistency-sufficient** (of the bit-wrong steps, how often the
    split-brain verdict is still right -- the *decision payoff*; high here because the constrained
    decoder leaves only consistency-invisible in-flight errors, the ED2-learned mirror read through
    the other oracle), and **full-state wrong** (the real model's overall teacher-forced error).
  - **right** -- the **consult-fact ratio**: the consistency answer (per-object converged/split
    view) as a fraction of the full state (replicas + in-flight + partition/crash/clock) -- how
    much cheaper the decision consult is.

The H12 reading on the real model: the consistency oracle is *redundant* as a verifier yet a
**cheaper, decision-sufficient** consult for the question an SRE/defender actually asks -- the
tiered-oracle thesis confirmed on the model that actually exists.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed6_two_oracle_learned import ED6TwoOracleLearnedResult

_BAR = "#9467bd"  # the learned-arm colour (distinct from the synthetic mode-split reds/blues)


def plot_ed6_two_oracle_learned(
    result: ED6TwoOracleLearnedResult, path: str | Path
) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, v = result.cell, result.verdict
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})

    rates = ("non_redundant_rate", "consistency_sufficient_rate", "full_wrong_rate")
    labels = ("non-redundant\n(H12: ≈0)", "consistency-sufficient\n(decision payoff)",
              "full-state\nwrong")
    xs = list(range(len(rates)))
    means = [c[m] for m in rates]
    yerr = [[c[m] - c[f"{m}_lo"] for m in rates], [c[f"{m}_hi"] - c[m] for m in rates]]
    ax_l.bar(xs, means, 0.55, yerr=yerr, capsize=5, color=_BAR, label="learned $M_\\theta$")
    for x, rate, m in zip(xs, rates, means, strict=True):
        ax_l.annotate(f"{m:.2f}", (x, c[f"{rate}_hi"] + 0.02), ha="center", va="bottom", fontsize=9)
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(labels, fontsize=9)
    ax_l.set_ylim(0, 1.12)
    ax_l.set_title("consistency second-oracle vs full bit-exact state (real $M_\\theta$)")
    ax_l.legend(loc="upper right", fontsize=8)
    ax_l.grid(True, axis="y", alpha=0.3)

    ratio = c["consult_fact_ratio"]
    ax_r.bar([0, 1], [1.0, ratio], color=["#c5b0d5", "#2ca02c"])
    ax_r.annotate(f"{ratio:.2f}\n(~{v['cheaper_factor']:.1f}× cheaper)" if ratio else "0",
                  (1, ratio), ha="center", va="bottom", fontsize=9)
    ax_r.set_xticks([0, 1])
    ax_r.set_xticklabels(["full state\n(replicas+inflight+medium)", "consistency\nview"],
                         fontsize=8)
    ax_r.set_ylim(0, 1.1)
    ax_r.set_ylabel("consult facts (fraction of full)")
    ax_r.set_title("the decision consult is far cheaper")
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED6 / H12 (learned $M_\\theta$) — the consistency oracle is redundant as a "
                 "verifier but decision-sufficient and cheaper (SPEC-7)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
