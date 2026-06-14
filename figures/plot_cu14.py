"""Plot the CU14 figure (SPEC-22 H107): the defended incident -- the whole stack on one scenario.

Two panels from a :class:`~verisim.acd.incident_response.CU14Result`:

  - **left -- the all-good corner.** Each defense as a point in (cost, breach) space; the marker is
    filled green when it completes the mission, hollow red when it abandons it. Undefended is cheap
    but breaches; paranoid is safe and cheap but off-mission (hollow); full oracle is safe and
    on-mission but far to the right (expensive); structure is the only point that is low and cheap
    and cheap -- the all-good corner.
  - **right -- the incident playback.** The same representative incident replayed under two defenses
    (undefended, structure): a timeline of its ``connect`` steps, work connects in blue, the
    crown-jewel lure marked red (undefended walks it = breach) vs green (structure catches it).
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.incident_response import CU14Result, StepRecord


def _connect_steps(steps: tuple[StepRecord, ...]) -> list[StepRecord]:
    return [s for s in steps if s.is_connect and s.dst_class in ("jewel", "work")]


def plot_cu14(result: CU14Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.4, 5.0))

    # left: the all-good corner -- (cost, breach), filled-green if on-mission else hollow-red
    led = {x.defense: x for x in result.ledgers}
    xmax = max(led["full_oracle"].mean_calls * 1.15, 8.0)
    # leader-line callouts placed clear of the cluster at the origin
    callout = {
        "undefended": ("undefended\nbreaches the mission", 0.30 * xmax, 0.86, "#d62728"),
        "paranoid": ("paranoid\nsafe but off-mission", 0.16 * xmax, 0.42, "#d62728"),
        "structure": ("structure  ✓\nsafe · on-mission · cheap", 0.30 * xmax, 0.24, "#2ca02c"),
        "full_oracle": ("full oracle\nsafe but 12× the cost", 0.74 * xmax, 0.30, "#2ca02c"),
    }
    for d, (text, tx, ty, color) in callout.items():
        c = led[d]
        on_mission = c.completion_rate >= 0.95
        ax_l.scatter([c.mean_calls], [c.breach_rate], s=300,
                     facecolor=color if on_mission else "none", edgecolor=color, linewidths=2.4,
                     zorder=3)
        ax_l.annotate(text, xy=(c.mean_calls, c.breach_rate), xytext=(tx, ty), ha="center",
                      fontsize=8.4, color=color,
                      fontweight="bold" if d == "structure" else "normal",
                      arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1})
    ax_l.set_xlabel("mean oracle calls per incident  (the cost)")
    ax_l.set_ylabel("breach rate  (exfiltrated at least once)")
    ax_l.set_title(f"only structure is safe, on-mission, and cheap  "
                   f"({result.n_episodes} incidents, real M_θ)", fontsize=9.2)
    ax_l.set_ylim(-0.06, 1.06)
    ax_l.set_xlim(-2.5, xmax)

    # right: the incident playback -- same actions, two defenses. A jewel connect is only a breach
    # if the oracle says it truly opens a flow; jewel connects that open nothing are benign (grey).
    u = _connect_steps(result.playback_undefended)
    s = _connect_steps(result.playback_structure)
    for steps, y in [(u, 1.0), (s, 0.0)]:
        for st in steps:
            if st.dst_class == "work":
                ax_r.scatter([st.step], [y], marker="o", s=70, color="#1f77b4", zorder=3)
            elif st.oracle_exfil:  # the true crown-jewel lure
                if st.breach:
                    ax_r.scatter([st.step], [y], marker="X", s=200, color="#d62728", zorder=5)
                else:
                    ax_r.scatter([st.step], [y], marker="P", s=200, color="#2ca02c", zorder=5)
            else:  # connect to a jewel host that opened no flow -- benign; structure still checks
                ax_r.scatter([st.step], [y], marker=".", s=55, color="#999999", zorder=2)
    ax_r.set_yticks([0.0, 1.0])
    ax_r.set_yticklabels(["structure", "undefended"], fontsize=9)
    ax_r.set_ylim(-0.6, 1.6)
    ax_r.set_xlabel(f"step in incident #{result.playback_index}")
    ax_r.set_title("the same lure: undefended walks it (X), structure catches it (+)", fontsize=9.2)
    from matplotlib.lines import Line2D
    ax_r.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", ms=9,
               label="work connect (mission)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#d62728", ms=11,
               label="exfil lure executed = BREACH"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="#2ca02c", ms=11,
               label="exfil lure verified + aborted"),
        Line2D([0], [0], marker=".", color="w", markerfacecolor="#999999", ms=12,
               label="jewel connect, no flow (benign)"),
    ], fontsize=7.6, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "CU14 / H107: the defended incident — restore work connectivity without exfiltrating; "
        "verifying the world's flow-genesis surface is safe, on-mission, and cheap",
        fontsize=9.0, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
