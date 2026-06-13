"""Plot the CU8 figure (SPEC-22 H101): the drift asymmetry -- world models hide danger by omission.

Two panels from a :class:`~verisim.acd.drift_asymmetry.CU8Result`:

  - **left -- the asymmetry.** For the danger (protected) and benign (work) hosts, the count of
    **omissions** (the model missed a real flow) versus **hallucinations** (the model invented one).
    Omissions dominate, and on the protected hosts the hallucination bar is ~0 -- the model hides
    exfil it never invents.
  - **right -- the safety consequence.** The gate's two error sources: the **missed-danger** source
    (omitted protected flows) versus the **false-alarm** source (hallucinated protected flows).
    Drift concentrates in the catastrophic missed-danger cell, which is why verification is
    load-bearing for safety (and why CU5-net's utility axis never moved).
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.drift_asymmetry import CU8Result


def plot_cu8(result: CU8Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    classes = ["protected\n(danger)", "work\n(benign)"]
    omitted = [result.omitted["protected"], result.omitted["work"]]
    hallucinated = [result.hallucinated["protected"], result.hallucinated["work"]]
    x = range(len(classes))
    width = 0.38
    ax_l.bar([i - width / 2 for i in x], omitted, width, color="#d62728",
             label="omission (missed a real flow)")
    ax_l.bar([i + width / 2 for i in x], hallucinated, width, color="#1f77b4",
             label="hallucination (invented a flow)")
    for i, (o, h) in enumerate(zip(omitted, hallucinated, strict=True)):
        ax_l.text(i - width / 2, o + 1, str(o), ha="center", fontsize=9, fontweight="bold")
        ax_l.text(i + width / 2, h + 1, str(h), ha="center", fontsize=9)
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels(classes)
    ax_l.set_ylabel("prediction errors over the battery")
    ax_l.set_ylim(0, max(omitted) * 1.18)
    ax_l.set_title(f"drift is omission-biased: the model hides flows, it does not invent them\n"
                   f"(teacher-forced, {result.n_steps} steps on a real trained M_θ)", fontsize=9.0)
    ax_l.legend(fontsize=8.6, loc="upper left")

    miss = result.omitted["protected"]
    false_alarm = result.hallucinated["protected"]
    bars = ax_r.bar(["missed-danger source\n(omitted exfil flow)",
                     "false-alarm source\n(hallucinated exfil flow)"],
                    [miss, false_alarm], color=["#d62728", "#1f77b4"])
    for b, v in zip(bars, [miss, false_alarm], strict=True):
        ax_r.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", fontsize=10,
                  fontweight="bold")
    ax_r.set_ylabel("exfil-prediction errors (protected hosts)")
    ax_r.set_title(f"drift concentrates in the catastrophic cell\n"
                   f"(the model foresaw only {result.recall('protected'):.0%} of true exfil flows)",
                   fontsize=9.2)

    fig.suptitle(
        "CU8 / H101: world models hide danger by OMISSION — drift inflates the missed-danger cell, "
        "not the false-alarm one; this is why the safety gate needs the oracle",
        fontsize=9.4, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
