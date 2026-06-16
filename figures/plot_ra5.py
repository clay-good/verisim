"""Plot the RA5 figure (SPEC-22 H137): the 4-arm head-to-head -- oracle gate vs shipped guardrails.

The centerpiece. Two panels from a :class:`~verisim.realagent.head_to_head.HeadToHeadResult`:

  - **left -- adversarial missed-harm by arm.** Under an adaptive prompt-injection adversary, the
    learned model guardrail (the SafePred proxy) and the permission denylist (the Claude-Code proxy)
    are both breached (1.0) -- a better model is no safer (CU4), and a denylist cannot enumerate
    every realizing action -- while the oracle coverage gate is un-gameable (0.0).
  - **right -- safety vs cost.** Each arm in (oracle calls per task, adversarial missed-harm) space.
    The free guardrails sit at zero cost but full breach; the oracle coverage gate is the only arm
    in the safe corner, at ~1 oracle call per task. (Utility is 1.0 for every arm here -- the danger
    surface is disjoint from the benign work -- so safety and cost are the live axes.)
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.head_to_head import HeadToHeadResult

_ORDER = ("none", "model_guardrail", "permission", "oracle_coverage")
_SHORT = {
    "none": "no gate",
    "model_guardrail": "learned\nguardrail\n(SafePred)",
    "permission": "permission\ndenylist\n(Claude-Code)",
    "oracle_coverage": "oracle\ncoverage\n(verisim)",
}
_COLORS = {
    "none": "#999", "model_guardrail": "#c44", "permission": "#e80", "oracle_coverage": "#2a7",
}


def plot_ra5(result: HeadToHeadResult, path: str | Path) -> Path:  # pragma: no cover - local plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {a.arm: a for a in result.arms}
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: adversarial missed-harm by arm -----------------------------------------------------
    x = range(len(_ORDER))
    adv = [by[a].adversarial_missed_harm for a in _ORDER]
    ax_l.bar(x, adv, color=[_COLORS[a] for a in _ORDER], width=0.62)
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels([_SHORT[a] for a in _ORDER], fontsize=8)
    ax_l.set_ylabel("adversarial missed-harm  (breach)")
    ax_l.set_ylim(0, 1.18)
    ax_l.set_title("Under an adaptive adversary: only the oracle gate holds")
    for i in range(len(_ORDER)):
        ax_l.text(i, adv[i] + 0.02, f"{adv[i]:.2f}", ha="center", fontsize=10)

    # --- right: safety vs cost --------------------------------------------------------------------
    for a in _ORDER:
        arm = by[a]
        ax_r.scatter(arm.mean_consult_cost, arm.adversarial_missed_harm, s=170,
                     color=_COLORS[a], zorder=3)
        ax_r.annotate(_SHORT[a].replace("\n", " "),
                      (arm.mean_consult_cost, arm.adversarial_missed_harm),
                      textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax_r.set_xlabel("oracle calls per task  (cost)")
    ax_r.set_ylabel("adversarial missed-harm  (breach)")
    ax_r.set_ylim(-0.08, 1.12)
    ax_r.set_xlim(-0.3, max(1.5, max(by[a].mean_consult_cost for a in _ORDER) + 0.5))
    ax_r.set_title("The oracle gate is the only one in the safe corner")
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        "RA5 / H137 -- the 4-arm head-to-head: the learned guardrail (SafePred) and the permission "
        "denylist (Claude-Code) are adversarially breached; only the oracle gate is un-gameable",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
