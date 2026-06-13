"""Plot the CU7 figure (SPEC-22 §5, H99): verify-before-commit -- where you verify beats how much.

Two panels from a :class:`~verisim.acd.closed_loop_verify.CU7Result`:

  - **left -- the cost/harm frontier.** Harm rate vs mean oracle calls per goal. The budgeted
    replanner (CU6) traces a frontier from the free agent (0 calls, high harm) up to full
    verification (zero harm, the most calls). Verify-before-commit is a single star that reaches the
    **zero-harm guarantee** far to the left -- below the frontier -- because it spends every call at
    the commit point. The dashed guide marks the cost each strategy pays to reach zero harm.
  - **right -- why it is cheaper (the mechanism).** A full-verification agent's calls split into the
    ones that matter (verifying the route it *commits* to -- the model's "yes") and the ones that
    are **wasted** (verifying routes the model already calls dangerous -- a "no" leads to an abort,
    which can never cause harm). Verify-before-commit keeps only the first slab. The wasted slab is
    the cost CU7 removes, and it grows with how adversarial the environment is.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.closed_loop_verify import CU7Result


def plot_cu7(result: CU7Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    calls = [c.calls for c in result.budgeted]
    harm = [c.harm_rate for c in result.budgeted]
    ax_l.plot(calls, harm, "-o", color="#9467bd", lw=2, ms=6,
              label="budgeted replanner (CU6, sweep ρ)")
    ax_l.scatter([result.vbc.calls], [result.vbc.harm_rate], s=320, marker="*", color="#2ca02c",
                 edgecolors="white", linewidths=1.2, zorder=5, label="verify-before-commit")
    ax_l.scatter([result.full_verify.calls], [result.full_verify.harm_rate], s=130, marker="s",
                 color="#d62728", edgecolors="white", linewidths=1.0, zorder=5,
                 label="full-verify (verify everything)")
    ax_l.axhline(0.0, color="#888", ls=":", lw=1.0)
    ax_l.annotate(
        f"zero-harm guarantee\nat {result.vbc.calls:.1f} calls vs {result.full_verify.calls:.1f}",
        xy=(result.vbc.calls, 0.0), xytext=(result.vbc.calls + 0.25, 0.06), fontsize=8.6,
        arrowprops={"arrowstyle": "->", "color": "#2ca02c"})
    ax_l.set_xlabel("mean oracle calls per goal  (the cost)")
    ax_l.set_ylabel("harm rate  (executed a dangerous route)")
    ax_l.set_title(f"verify at the commit point: zero harm, far cheaper  (φ={result.phi})",
                   fontsize=9.2)
    ax_l.set_ylim(-0.02, max(harm) * 1.08)
    ax_l.legend(fontsize=8.4, loc="upper right")

    full = result.full_verify
    needed_full = full.calls - full.wasted_calls
    labels = ["full-verify\n(verify everything)", "verify-before-commit\n(verify the commit)"]
    needed = [needed_full, result.vbc.calls]
    wasted = [full.wasted_calls, 0.0]
    ax_r.bar(labels, needed, color="#2ca02c", label="necessary (verify the model's 'yes')")
    ax_r.bar(labels, wasted, bottom=needed, color="#d62728", alpha=0.55,
             label="wasted (verify a 'no' that can't cause harm)")
    for i, (n, w) in enumerate(zip(needed, wasted, strict=True)):
        ax_r.text(i, n + w + 0.04, f"{n + w:.2f} calls", ha="center", fontsize=9, fontweight="bold")
    wasted_frac = full.wasted_calls / full.calls if full.calls else 0.0
    ax_r.set_ylabel("mean oracle calls per goal")
    ax_r.set_title(f"{wasted_frac:.0%} of full verification is wasted on 'no' decisions",
                   fontsize=9.5)
    ax_r.legend(fontsize=8.4, loc="upper right")

    fig.suptitle(
        "CU7 / H99: verify-before-commit — harm only happens at the commit, so verify the model's "
        "'yes' and trust its 'no'; the zero-harm guarantee at a fraction of the oracle cost",
        fontsize=9.3, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
