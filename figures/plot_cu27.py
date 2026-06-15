"""Plot the CU27 figure (SPEC-22 H120): the reversibility boundary -- *when* to verify, not *what*.

Two panels from a :class:`~verisim.acd.reversibility_boundary.CU27Result`:

  - **left -- the two danger classes under four policies.** Grouped bars of adversarial breach by
    policy, split reversible (exposure, a posture you can roll back) vs irreversible (an exfil send
    that already left). Verify-after-commit is the green safe bar on the reversible class at ZERO
    oracle previews (free + model-free + un-gameable) but the red full bar on the irreversible class
    (a send cannot be undone). Routing is safe on both; the free omitter gate is unsafe on both.
  - **right -- the cost law.** Sweeping the irreversible fraction ``f`` of the action space: the
    routed policy's before-commit oracle cost rises *linearly* with ``f`` (zero on the reversible
    half), the after-commit-everywhere residual breach *also* rises with ``f`` (it cannot undo the
    irreversible part), and verify-everything overpays a constant full cost. The price of
    trustworthy world-modeling is exactly the irreversibility you face.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.reversibility_boundary import CU27Result

_POLICY_LABEL = {
    "after_commit": "verify-after-commit\n(free, model-free)",
    "before_commit_free": "before-commit, free\n(omitter gate)",
    "before_commit_oracle": "before-commit, verify all\n(treat all irreversible)",
    "routed": "routed by reversibility",
}
_ORDER = ("after_commit", "before_commit_free", "before_commit_oracle", "routed")


def plot_cu27(result: CU27Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    # left: adversarial breach by policy x danger class (grouped bars)
    rev = [result.cell(p, "reversible") for p in _ORDER]
    irr = [result.cell(p, "irreversible") for p in _ORDER]
    x = list(range(len(_ORDER)))
    w = 0.38
    bars_rev = ax_l.bar(
        [xi - w / 2 for xi in x], [c.adversarial_breach for c in rev], w,
        color="#2ca02c", label="reversible (exposure posture)",
    )
    bars_irr = ax_l.bar(
        [xi + w / 2 for xi in x], [c.adversarial_breach for c in irr], w,
        color="#d62728", label="irreversible (exfil send)",
    )
    for c, b in zip(rev, bars_rev, strict=True):
        tag = "0 oracle" if c.mean_oracle_calls <= 1e-9 else f"{c.mean_oracle_calls:.1f} oracle"
        ax_l.annotate(tag, xy=(b.get_x() + b.get_width() / 2, b.get_height() + 0.02),
                      ha="center", fontsize=7.0, color="#2ca02c")
    for c, b in zip(irr, bars_irr, strict=True):
        tag = "0 oracle" if c.mean_oracle_calls <= 1e-9 else f"{c.mean_oracle_calls:.1f} oracle"
        ax_l.annotate(tag, xy=(b.get_x() + b.get_width() / 2, b.get_height() + 0.02),
                      ha="center", fontsize=7.0, color="#d62728")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([_POLICY_LABEL[p] for p in _ORDER], fontsize=7.4)
    ax_l.set_ylabel("adversarial (worst-case) breach rate")
    ax_l.set_ylim(0, 1.18)
    ax_l.set_title("reversible danger is safe model-free via verify-after-commit (0 previews);\n"
                   "irreversible needs verify-before-commit -- a send cannot be rolled back",
                   fontsize=9.0)
    ax_l.legend(fontsize=8.0, loc="upper center")

    # right: the cost law over the irreversible fraction f
    law = result.cost_law
    fs = [p.irreversible_fraction for p in law]
    ax_r.plot(fs, [p.routed_oracle_calls for p in law], "-o", color="#1f77b4", lw=2.2, ms=8,
              label="routed: before-commit oracle calls")
    ax_r.plot(fs, [p.verify_all_oracle_calls for p in law], "--s", color="#888", lw=1.8, ms=7,
              label=f"verify-everything ({result.horizon} calls, flat)")
    ax_r.set_xlabel("irreversible fraction  f  of the action space")
    ax_r.set_ylabel("mean before-commit oracle calls / deployment", color="#1f77b4")
    ax_r.tick_params(axis="y", labelcolor="#1f77b4")
    ax_r.set_title("the cost law: the price of trustworthy world-modeling\n"
                   "is exactly the irreversibility you face", fontsize=9.0)
    rt_irr_calls = result.cell("routed", "irreversible").mean_oracle_calls
    ax_r.annotate(
        f"routed = {rt_irr_calls:.1f} calls at f=1\n"
        f"vs {result.horizon} verify-all "
        f"({result.horizon / max(rt_irr_calls, 1e-9):.0f}x)",
        xy=(1.0, law[-1].routed_oracle_calls), xytext=(0.42, result.horizon * 0.55),
        fontsize=8.0, color="#1f77b4", arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.1),
    )

    ax_r2 = ax_r.twinx()
    ax_r2.plot(fs, [p.after_commit_breach for p in law], "-^", color="#d62728", lw=2.0, ms=8,
               label="after-commit-everywhere: residual breach")
    ax_r2.set_ylabel("after-commit residual breach rate", color="#d62728")
    ax_r2.tick_params(axis="y", labelcolor="#d62728")
    ax_r2.set_ylim(-0.03, 1.05)
    lines_l, labels_l = ax_r.get_legend_handles_labels()
    lines_r, labels_r = ax_r2.get_legend_handles_labels()
    ax_r.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7.6, loc="upper left")

    fig.suptitle(
        "CU27 / H120: the reversibility boundary -- a new axis orthogonal to the targeting arc. "
        "Reversible dangers are safe MODEL-FREE via verify-after-commit (0 previews);\nmodel "
        "faithfulness / the targeting machinery is load-bearing ONLY on the irreversible slice  "
        f"({result.n_reversible} reversible + {result.n_irreversible} irreversible, "
        f"horizon {result.horizon})",
        fontsize=8.7, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
