"""Plot the CU34 figure (SPEC-22 H127): the footprintless danger -- host confidentiality.

Two panels from a :class:`~verisim.acd.footprintless_targeting.CU34Result`:

  - **left -- after-the-fact detection is blind to the footprintless danger.** For each host CIA
    leg, the catch rate of an after-the-fact state-diff detector (it watches protected resources for
    a change) vs the safety of the before-commit covering target. The detector catches integrity and
    availability (they mutate state) but is at ZERO on confidentiality -- a secret read changes
    nothing, so there is no footprint to detect. The before-commit target prevents all three.
  - **right -- the confidentiality frontier: verify the read surface before commit.** Adversarial
    breach vs oracle calls. The uniform schedule is pinned at 1.0 until the full oracle (the knee is
    a mirage); the model self-targets to failure; the CU16 write target carried over leaks (a read
    is not a write, covers=False); only the derived read-to-secret-fd target reaches zero breach,
    cheaply and un-gameably.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.footprintless_targeting import CU34Result


def plot_cu34(result: CU34Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.6, 5.0))

    # --- left: after-the-fact detection vs before-commit target, per CIA leg ----------------------
    legs = ["integrity\n(write)", "availability\n(kill)", "confidentiality\n(read)"]
    detect = [c.after_the_fact_catch_rate for c in result.contrasts]
    prevent = [1.0 - c.before_commit_adversarial for c in result.contrasts]
    x = range(len(legs))
    w = 0.38
    ax_l.bar([i - w / 2 for i in x], detect, w, color="#c44",
             label="after-the-fact detector (state diff)")
    ax_l.bar([i + w / 2 for i in x], prevent, w, color="#2a7",
             label="before-commit target (verify the action)")
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels(legs)
    ax_l.set_ylabel("handled rate  (detected / prevented)")
    ax_l.set_ylim(0, 1.18)
    ax_l.set_title("After-the-fact detection is blind to the footprintless danger")
    ax_l.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax_l.annotate("no footprint:\nnothing to detect", xy=(2 - w / 2, 0.0),
                  xytext=(1.45, 0.42), fontsize=8.5, color="#c44",
                  ha="center", arrowprops=dict(arrowstyle="->", color="#c44"))
    for i, (d, p) in enumerate(zip(detect, prevent, strict=True)):
        ax_l.text(i - w / 2, d + 0.02, f"{d:.2f}", ha="center", fontsize=8, color="#c44")
        ax_l.text(i + w / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8, color="#2a7")

    # --- right: the confidentiality frontier (adversarial breach vs calls) ------------------------
    arm = result.confidentiality
    uni_calls = [c.mean_calls for c in arm.uniform]
    uni_adv = [c.adversarial_breach for c in arm.uniform]
    ax_r.plot(uni_calls, uni_adv, "-o", color="#88a", label="uniform (the knee is a mirage)",
              markersize=4)
    ax_r.scatter([arm.model.mean_calls], [arm.model.adversarial_breach], marker="x", s=90,
                 color="#a33", zorder=5, label="model self-targeting (fails)")
    if arm.shortcut is not None:
        ax_r.scatter([arm.shortcut.mean_calls], [arm.shortcut.adversarial_breach], marker="P",
                     s=95, color="#d62", zorder=5,
                     label="CU16 write target carried over (leaks, covers=False)")
    ax_r.scatter([arm.full_oracle.mean_calls], [arm.full_oracle.adversarial_breach], marker="s",
                 s=70, color="#555", zorder=5, label="full oracle (48 calls)")
    ax_r.scatter([arm.target.mean_calls], [arm.target.adversarial_breach], marker="*", s=320,
                 color="#2a7", zorder=6,
                 label="read-to-secret-fd target (DERIVED): safe, un-gameable")
    saving = (arm.full_oracle.mean_calls / arm.target.mean_calls
              if arm.target.mean_calls > 0 else float("inf"))
    ax_r.annotate(f"{arm.target.mean_calls:.1f} calls\n{saving:.0f}x cheaper",
                  xy=(arm.target.mean_calls, 0.0), xytext=(arm.target.mean_calls + 7, 0.18),
                  fontsize=8.5, color="#2a7",
                  arrowprops=dict(arrowstyle="->", color="#2a7"))
    ax_r.set_xlabel("oracle calls per deployment")
    ax_r.set_ylabel("adversarial breach rate")
    ax_r.set_ylim(-0.05, 1.12)
    ax_r.set_title("Confidentiality: verify the read surface before commit")
    ax_r.legend(loc="center right", fontsize=7.6, framealpha=0.92)

    fig.suptitle(
        "CU34 / H127 -- the footprintless danger: a secret read mutates nothing, so it must be "
        "verified before commit",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
