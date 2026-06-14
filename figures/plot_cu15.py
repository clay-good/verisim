"""Plot the CU15 figure (SPEC-22 H108): the verification-exhaustion attack -- the cost axis.

Two panels from a :class:`~verisim.acd.verification_exhaustion.CU15Result`, sharing the attacker's
saturation on the x-axis so the symmetry is the point:

  - **left -- the safety axis (breach vs saturation).** Structure is flat at 0 (its safety is
    immovable -- every poisoned ``connect``-to-jewel is oracle-blocked); uniform and the free agent
    rise as the attacker floods off-clock exfils the omitting model misses (uniform's safety is
    gameable). Full oracle flat at 0.
  - **right -- the cost axis (oracle calls vs saturation).** Now the picture inverts: structure's
    cost climbs toward the full oracle as the flood grows (its cost is gameable), while uniform's
    clock-keyed cost stays flat (immovable). Structure stays under the full-oracle line at every
    saturation -- the shaded band is the discount the attacker can erase but never overdraw.

Each sub-oracle schedule is flat in one panel and rising in the other; only the full oracle is
flat-and-safe in both, at the maximum price. The defender's lesson: prefer the schedule whose
movable axis is the bill, not the breach.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.verification_exhaustion import CU15Result, _series


def plot_cu15(result: CU15Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    free = _series(result, "free (no verification)")
    uniform = _series(result, "uniform ρ=0.5")
    structure = _series(result, "structure (crown-jewel)")
    full = _series(result, "full oracle")
    sat = [c.saturation for c in structure]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    # left: the safety axis -- breach vs saturation
    ax_l.plot(sat, [c.breach_rate for c in free], "-o", color="#888888", lw=1.8, ms=5,
              label="free (no verification)")
    ax_l.plot(sat, [c.breach_rate for c in uniform], "-s", color="#7f7fbf", lw=2.0, ms=6,
              label="uniform ρ=0.5 (safety gameable)")
    ax_l.plot(sat, [c.breach_rate for c in structure], "-*", color="#2ca02c", lw=2.4, ms=12,
              label="structure (safety immovable)")
    ax_l.plot(sat, [c.breach_rate for c in full], "--", color="#222222", lw=1.6,
              label="full oracle")
    ax_l.set_xlabel("attacker saturation  s  (fraction of steps poisoned)")
    ax_l.set_ylabel("breach rate over the deployment")
    ax_l.set_ylim(-0.04, 1.05)
    ax_l.set_title("the safety axis: structure flat at 0; uniform rises (gameable)", fontsize=9.0)
    ax_l.legend(fontsize=8.0, loc="center left")

    # right: the cost axis -- oracle calls vs saturation
    s_calls = [c.mean_calls for c in structure]
    f_calls = [c.mean_calls for c in full]
    ax_r.fill_between(sat, s_calls, f_calls, color="#2ca02c", alpha=0.10,
                      label="the discount the attacker erases")
    ax_r.plot(sat, [c.mean_calls for c in uniform], "-s", color="#7f7fbf", lw=2.0, ms=6,
              label="uniform ρ=0.5 (cost immovable)")
    ax_r.plot(sat, s_calls, "-*", color="#2ca02c", lw=2.4, ms=12,
              label="structure (cost gameable, bounded)")
    ax_r.plot(sat, f_calls, "--", color="#222222", lw=1.6, label="full oracle (the cost ceiling)")
    ax_r.annotate(
        f"{s_calls[0]:.1f} calls\n(random)", xy=(sat[0], s_calls[0]),
        xytext=(sat[0] + 0.08, s_calls[0] + 8), fontsize=7.8, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.9),
    )
    ax_r.annotate(
        f"-> {s_calls[-1]:.1f} = full oracle", xy=(sat[-1], s_calls[-1]),
        xytext=(sat[-1] - 0.52, s_calls[-1] - 12), fontsize=7.8, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.9),
    )
    ax_r.set_xlabel("attacker saturation  s  (fraction of steps poisoned)")
    ax_r.set_ylabel("mean oracle calls per deployment")
    ax_r.set_ylim(0, result.horizon * 1.08)
    ax_r.set_title("the cost axis: structure rises to (never past) the oracle; uniform flat",
                   fontsize=9.0)
    ax_r.legend(fontsize=8.0, loc="center left")

    fig.suptitle(
        "CU15 / H108: the verification-exhaustion attack — an adversary moves exactly one axis "
        "of a sub-oracle schedule\n(structure's cost, a bill ≤ the full oracle and self-limiting; "
        f"uniform's safety, a breach); only the full oracle is immovable on both  "
        f"({result.n_episodes} deployments, real M_θ)",
        fontsize=9.0, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
