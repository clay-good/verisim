"""Plot the ED2-smart figure (SPEC-7 §8.1, §10 ED2; DS7): the π_c smart-when policy comparison.

One panel: grouped bars of faithful horizon per consultation policy (`fixed` | `uncertainty` |
`drift`) across a few interior budgets ``ρ``, with bootstrap-CI error bars. All three policies spend
*exactly* the same budget at each ``ρ`` (the runner's spend-down backstop), so the bars isolate
*where* a policy spends its consults, not *how much* -- the H9 question.

The figure makes the **flat-arm smart-π_c verdict** visible: the learned model's decode-entropy
signal does (or does not) lift horizon over `fixed` at equal budget. The pre-registered expectation
is the standing H2/H9 negative carried into the distributed world -- per-step decode entropy is a
decode-time artifact, not a calibrated belief -- which localizes the smart-π_c lever to the
(deferred) structured `M_θ`'s RSSM belief variance, exactly as the host's EH2 found.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed2_smart import ED2SmartResult

_COLORS = {
    "fixed": "#d62728",
    "uncertainty": "#2ca02c",
    "drift": "#1f77b4",
}


def plot_ed2_smart(result: ED2SmartResult, path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = list(dict.fromkeys(c["rho"] for c in result.cells))
    policies = list(dict.fromkeys(c["policy"] for c in result.cells))
    by = {(c["policy"], c["rho"]): c for c in result.cells}

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.8))
    n = len(policies)
    width = 0.8 / n
    x = range(len(rhos))
    for i, policy in enumerate(policies):
        offs = [xi + (i - (n - 1) / 2) * width for xi in x]
        ys = [by[(policy, r)]["h_eps"] for r in rhos]
        lo = [by[(policy, r)]["h_eps"] - by[(policy, r)]["ci_lo"] for r in rhos]
        hi = [by[(policy, r)]["ci_hi"] - by[(policy, r)]["h_eps"] for r in rhos]
        ax.bar(offs, ys, width, label=policy, color=_COLORS.get(policy, "#888"),
               yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1, "alpha": 0.6})

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"ρ={r:g}" for r in rhos])
    ax.set_xlabel("consultation budget  ρ  (all policies spend the same at each ρ)")
    ax.set_ylabel("faithful horizon  H_ε  (steps)")
    ax.legend(title="π_c policy", fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    verdict_by_rho = {v["rho"]: v for v in result.verdict}
    any_win = any(v["smart_wins"] for v in result.verdict)
    tag = "smart-π_c beats fixed" if any_win else "smart-π_c does NOT beat fixed (flat-arm H9 null)"
    lifts = "  ".join(
        f"ρ={r:g}:{verdict_by_rho[r]['lift']:.2f}×" for r in rhos if r in verdict_by_rho
    )
    ax.set_title(f"ED2 π_c smart-when: {tag}\nbest-smart/fixed horizon ratio — {lifts}",
                 fontsize=10)

    fig.suptitle("ED2-smart — does entropy-gated consultation beat fixed at equal budget? "
                 "(H9, SPEC-7)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
