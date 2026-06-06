"""Plot the ED8 figure (SPEC-7 §3.2, DS0 incr 2): the OCC commit/abort frontier under contention.

Two panels from one :class:`~verisim.experiments.ed8.ED8Result`:

  - **left -- the commit-rate frontier vs the occupancy law.** Measured commit rate (the share of
    ``K`` concurrent transactions that commit) with bootstrap-CI whiskers, overlaid on the
    closed-form balls-in-bins prediction ``M·(1−(1−1/M)^K)/K``, as the contention dial ``M`` (number
    of objects) is swept. The measured points sit on the analytic curve: as objects multiply, the
    read-sets collide less and the first-committer-wins abort rate falls.
  - **right -- the commit/abort split.** Total commits vs conflicts per contention level (stacked),
    the operational read of the same data: at ``M=1`` (one hot object) exactly one of every batch
    commits and the rest abort; the aborts melt away as contention drops.

The headline annotation reports the max |measured − occupancy| gap (≈0, the semantics are exactly
right) and that Tier-B (the autonomous-actor system oracle) agrees with Tier-A on every scenario.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed8 import ED8Result


def plot_ed8(result: ED8Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result.rows
    objects = [r["objects"] for r in rows]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    # --- left: commit rate vs the occupancy law --------------------------------------------------
    measured = [r["commit_rate"] for r in rows]
    lo = [max(0.0, r["commit_rate"] - r["ci_lo"]) for r in rows]
    hi = [max(0.0, r["ci_hi"] - r["commit_rate"]) for r in rows]
    occ = [r["occupancy_rate"] for r in rows]
    ax_l.plot(objects, occ, color="#1f77b4", lw=2.2, marker="o",
              label="balls-in-bins occupancy  M(1−(1−1/M)^K)/K")
    ax_l.errorbar(objects, measured, yerr=[lo, hi], fmt="x", color="#d62728", capsize=4, lw=1.4,
                  ms=8, label="measured OCC commit rate (±95% CI)")
    ax_l.set_xlabel("objects  M  (contention dial — fewer = hotter)")
    ax_l.set_ylabel("commit rate  (committed / K transactions)")
    ax_l.set_ylim(0, 1.04)
    gap = max(abs(r["commit_rate"] - r["occupancy_rate"]) for r in rows)
    ax_l.set_title(f"ED8: OCC commit rate tracks the occupancy law (max gap = {gap:.3f})")
    ax_l.legend(loc="lower right", fontsize=8)
    ax_l.grid(True, alpha=0.3)

    # --- right: the commit/abort split ------------------------------------------------------------
    commits = [r["commits"] for r in rows]
    conflicts = [r["conflicts"] for r in rows]
    x = range(len(objects))
    ax_r.bar(x, commits, color="#2ca02c", alpha=0.85, label="committed")
    ax_r.bar(x, conflicts, bottom=commits, color="#d62728", alpha=0.8, label="aborted (conflict)")
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels([str(m) for m in objects])
    ax_r.set_xlabel("objects  M")
    ax_r.set_ylabel("transactions (summed over seeds)")
    ax_r.set_title("ED8: first-committer-wins — aborts melt as contention drops")
    ax_r.legend(loc="upper right", fontsize=8)

    all_agree = all(r["tier_b_agrees"] for r in rows)
    headline = (f"K concurrent transactions, OCC first-committer-wins  •  measured = occupancy "
                f"law  •  Tier-B agrees with Tier-A on every scenario = {all_agree}")
    fig.suptitle(headline, y=1.00, fontsize=9.5)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
