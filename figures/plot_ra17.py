"""Plot the RA17 figure (SPEC-22 H149): the oracle coverage gate as a Claude Code permission hook.

Two panels from the :func:`~verisim.realagent.claude_code_gate.run_gate_eval` results:

  - **left -- the two failure modes, by arm.** Grouped bars: benign approval fatigue (fraction of
    benign coding calls that prompt) and explicit missed-harm (fraction of realizing calls
    auto-allowed). The denylist status quo shows both (it over-prompts on a benign `rm` and leaks
    chmod/mv classes); the coverage hook shows neither.
  - **right -- the operating point.** Each arm in (benign approval burden, missed-harm) space. The
    no-gate arm is bottom-right (no prompts, all harm); the denylist is mid (some fatigue, some
    breach); the oracle coverage hook is in the (0, 0) corner (no benign fatigue, no missed harm),
    which is the §8 product operating point.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.claude_code_gate import ArmResult

_ORDER = ("allow_all", "permission_denylist", "oracle_coverage")
_SHORT = {
    "allow_all": "no gate\n(allow all)",
    "permission_denylist": "permission\ndenylist\n(status quo)",
    "oracle_coverage": "oracle\ncoverage hook\n(verisim)",
}
_COLORS = {"allow_all": "#999", "permission_denylist": "#e80", "oracle_coverage": "#2a7"}


def plot_ra17(results: list[ArmResult], path: str | Path) -> Path:  # pragma: no cover - local plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {r.arm: r for r in results}
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: the two failure modes, grouped bars ------------------------------------------------
    x = range(len(_ORDER))
    w = 0.38
    fatigue = [by[a].benign_prompt_rate for a in _ORDER]
    breach = [by[a].missed_harm_explicit for a in _ORDER]
    ax_l.bar([i - w / 2 for i in x], fatigue, width=w, color="#c69",
             label="benign approval fatigue")
    ax_l.bar([i + w / 2 for i in x], breach, width=w, color="#c44", label="explicit missed-harm")
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels([_SHORT[a] for a in _ORDER], fontsize=8)
    ax_l.set_ylabel("rate")
    ax_l.set_ylim(0, 1.15)
    ax_l.set_title("The denylist shows both failures; the coverage hook shows neither")
    ax_l.legend(fontsize=8, loc="upper center")
    for i in range(len(_ORDER)):
        ax_l.text(i - w / 2, fatigue[i] + 0.02, f"{fatigue[i]:.2f}", ha="center", fontsize=8)
        ax_l.text(i + w / 2, breach[i] + 0.02, f"{breach[i]:.2f}", ha="center", fontsize=8)

    # --- right: the operating point ---------------------------------------------------------------
    for a in _ORDER:
        r = by[a]
        ax_r.scatter(r.benign_prompt_rate, r.missed_harm_explicit, s=190,
                     color=_COLORS[a], zorder=3)
        ax_r.annotate(_SHORT[a].replace("\n", " "), (r.benign_prompt_rate, r.missed_harm_explicit),
                      textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax_r.scatter([0], [0], s=520, facecolors="none", edgecolors="#2a7", linewidths=1.6, zorder=2)
    ax_r.annotate("the §8 corner:\nno fatigue, no missed harm", (0, 0),
                  textcoords="offset points", xytext=(28, 24), fontsize=8, color="#2a7",
                  arrowprops=dict(arrowstyle="->", color="#2a7"))
    ax_r.set_xlabel("benign approval fatigue  (fraction of benign calls prompted)")
    ax_r.set_ylabel("explicit missed-harm  (realizing calls auto-allowed)")
    ax_r.set_xlim(-0.05, 0.6)
    ax_r.set_ylim(-0.08, 1.12)
    ax_r.set_title("The coverage hook is the only arm in the safe-and-quiet corner")
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        "RA17 / H149 -- the coverage gate as a Claude Code PreToolUse hook: fire the approval "
        "prompt on the covering surface, not a pattern list (paper §8)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
