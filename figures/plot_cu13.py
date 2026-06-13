"""Plot the CU13 figure (SPEC-22 H106): the false-alarm channel prices CU6, recall prices CU7.

Two panels from a :class:`~verisim.acd.closed_loop_replan_net.CU13Result`:

  - **left -- CU6's harm-amplification vs the false-alarm rate** (recall fixed at 0). The
    replanner's extra harm over a one-shot agent rises from zero with the model's *wrong* "no"s; the
    real trained ``M_θ`` (measured false-alarm ~0) anchors at the origin -- no amplification.
  - **right -- CU7's verify-before-commit saving vs the danger recall** (false-alarm fixed at 0).
    The full-verify / verify-before-commit cost ratio rises from 1x with the model's *right* "no"s;
    the real ``M_θ`` (measured recall ~0) anchors at the origin -- no saving.

The real model says "yes" to every route, so it sits at the origin of both dials: both CU6's
amplification and CU7's saving are artifacts of two-sided drift. (The danger does not vanish -- the
agent's one-shot harm is high either way -- only the *amplification* and the *saving* do.)
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.closed_loop_replan_net import CU13Result


def plot_cu13(result: CU13Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))
    real = result.real

    # left: CU6 -- harm-amplification vs the false-alarm rate (recall = 0)
    fx = [c.false_alarm_rate for c in result.fa_sweep]
    famp = [c.amplification for c in result.fa_sweep]
    ax_l.plot(fx, famp, "-o", color="#d62728", lw=2.0, ms=6, label="dial: synthetic net model")
    ax_l.axhline(0.0, color="#888", lw=1.0, ls=":")
    if real is not None:
        ax_l.plot([real.false_alarm_rate], [real.amplification], "*", color="#2ca02c", ms=20,
                  label=f"real M_θ (false-alarm {real.false_alarm_rate:.2f})", zorder=5)
        ax_l.annotate("amplification 0.000", (real.false_alarm_rate, real.amplification),
                      textcoords="offset points", xytext=(34, 6), fontsize=8.2, color="#2ca02c")
    ax_l.set_xlabel("model false-alarm rate  (wrong 'no's on safe routes)")
    ax_l.set_ylabel("harm-amplification  (replanner − one-shot harm)")
    ax_l.set_title("CU6: replanning amplifies harm only when the model false-alarms", fontsize=9.0)
    ax_l.legend(fontsize=8.2, loc="upper left")

    # right: CU7 -- verify-before-commit cost saving vs the danger recall (false-alarm = 0)
    rx = [c.recall_rate for c in result.recall_sweep]
    rsave = [c.cost_saving for c in result.recall_sweep]
    ax_r.plot(rx, rsave, "-o", color="#1f77b4", lw=2.0, ms=6, label="dial: synthetic net model")
    ax_r.axhline(1.0, color="#888", lw=1.0, ls=":")
    if real is not None:
        ax_r.plot([real.recall_rate], [real.cost_saving], "*", color="#2ca02c", ms=20,
                  label=f"real M_θ (recall {real.recall_rate:.2f})", zorder=5)
        ax_r.annotate("saving 1.00×", (real.recall_rate, real.cost_saving),
                      textcoords="offset points", xytext=(30, -14), fontsize=8.2, color="#2ca02c")
    ax_r.set_xlabel("model danger recall  (right 'no's on dangerous routes)")
    ax_r.set_ylabel("verify-before-commit saving  (full-verify / VBC calls)")
    ax_r.set_title("CU7: verify-before-commit saves cost only when the model recalls danger",
                   fontsize=9.0)
    ax_r.legend(fontsize=8.2, loc="upper left")

    one_shot = real.one_shot_harm if real is not None else 0.0
    fig.suptitle(
        "CU13 / H106: the false-alarm channel prices CU6, recall prices CU7 — a real "
        f"omission-biased M_θ says 'yes' to everything,\nso it shows neither amplification nor "
        f"saving (yet stays unsafe: one-shot harm {one_shot:.2f}, verify-before-commit still "
        f"zero-harm)  ({result.n_goals} goals, real M_θ)",
        fontsize=9.0, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
