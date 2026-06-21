"""Plot the RA25 figure (SPEC-22 H157): closing the mapped frontier, and the treadmill.

Two panels:
  - **left -- the fold closes the frontier.** Cumulative adversary reward vs oracle calls, the same
    neural policy against the base RA24 resolver vs the RA25 folded resolver. The folded curve is
    nearly flat: folding printf-escapes + pure rev/cut pipelines closes the entire mapped frontier
    on the twelve-mechanism grammar, so the adversary is left with nothing to find (sound).
  - **right -- the treadmill.** The pure-filter battery: how many filters the folder covers (FIRES)
    vs the unbounded ABSTAIN tail it does not, with the reversibility note -- every reversible
    ABSTAIN
    is already post-commit-diff safe, so the fold is usability, not safety.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.frontier_close import FrontierResult
from verisim.realagent.neural_proposer import RunResult


def plot(base: RunResult, folded: RunResult, fr: FrontierResult,
         path: str | Path) -> Path:  # pragma: no cover - needs matplotlib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: the adversary's reward collapses under the fold ------------------------------------
    calls = range(1, base.budget + 1)
    ax_l.plot(calls, base.reward_curve, color="#c33", lw=2,
              label=f"base RA24 resolver: {base.reward_per_call:.2f}/call")
    ax_l.plot(calls, folded.reward_curve, color="#2a7", lw=2,
              label=f"RA25 folded resolver: {folded.reward_per_call:.2f}/call")
    ax_l.set_xlabel("oracle calls")
    ax_l.set_ylabel("cumulative adversary reward (frontier)")
    ax_l.set_title("Folding closes the mapped frontier:\nthe adversary's reward collapses")
    ax_l.legend(fontsize=9, loc="upper left")
    ax_l.text(0.98, 0.05,
              f"silent miss: base {base.silent_miss} / folded {folded.silent_miss}  (sound)",
              transform=ax_l.transAxes, ha="right", va="bottom", fontsize=8,
              bbox={"boxstyle": "round", "fc": "#efe", "ec": "#9c9"})

    # --- right: the treadmill (covered vs the unbounded tail) ---------------------------------
    from verisim.realagent.frontier_close import treadmill_battery
    from verisim.realagent.shell_resolver import abstract_targets_protected as _resolve
    verdicts = [(name, _resolve(cmd, fold_filters=True)) for name, _c, cmd in treadmill_battery()]
    colors = ["#2a7" if v == "FIRES" else "#c93" for _n, v in verdicts]
    ax_r.bar(range(len(verdicts)), [1] * len(verdicts), color=colors)
    ax_r.set_xticks(range(len(verdicts)))
    ax_r.set_xticklabels([n for n, _v in verdicts], fontsize=8, rotation=30, ha="right")
    ax_r.set_yticks([])
    ax_r.set_title("The treadmill: folder covers what it was taught\n(green=FIRES), the tail stays "
                   "ABSTAIN (amber)")
    ax_r.text(0.5, 0.5,
              f"covered {fr.covered_after}/{fr.n_filters}\ntail (ABSTAIN): "
              f"{', '.join(fr.abstain_after)}\n\nall reversible -> post-commit diff\n"
              f"closes them with NO folder\n(fold = usability, diff = safety)",
              transform=ax_r.transAxes, ha="center", va="center", fontsize=9,
              bbox={"boxstyle": "round", "fc": "#eef", "ec": "#99c"})

    fig.suptitle("RA25: closing the mapped frontier is an enumeration treadmill; the post-commit "
                 "diff is the non-enumerative closer", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
