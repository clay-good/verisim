"""Plot the RA21 figure (SPEC-22 H152): the coverage gate on a real Claude Code corpus.

Reads the committed, privacy-safe ``bench/cc_corpus/summary.json`` (rates only, no raw commands) and
draws two panels:

  - **left -- the real approval burden.** Per arm, the fraction of real state-mutating tool calls
    escalated to a human. The coverage gate's prompts are the sparse on-surface density; the
    denylist's bar is split into the (rare) on-surface part and the (dominant) off-surface fatigue.
    The off-surface auto-approved majority is annotated.
  - **right -- harm coverage on the labeled arsenal.** Missed-harm per arm: the coverage gate 0.00,
    the denylist's enumeration-treadmill leak, allow-all 1.00.

Run: ``python figures/plot_ra21_cc_corpus.py`` (after ``bench/cc_corpus/extract.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

_ORDER = ("allow_all", "permission_denylist", "oracle_coverage")
_SHORT = {
    "allow_all": "no gate\n(allow all)",
    "permission_denylist": "permission\ndenylist\n(status quo)",
    "oracle_coverage": "oracle\ncoverage gate\n(verisim)",
}


def plot_ra21(summary: dict[str, object], path: str | Path) -> Path:  # pragma: no cover - plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(summary["n_calls"])  # type: ignore[arg-type]
    cov_prompt = float(summary["coverage_prompt_rate"])  # type: ignore[arg-type]
    deny_prompt = float(summary["denylist_prompt_rate"])  # type: ignore[arg-type]
    deny_fatigue = float(summary["denylist_fatigue_rate"])  # type: ignore[arg-type]
    deny_onsurface = deny_prompt - deny_fatigue
    harm = summary["harm_detail"]  # type: ignore[index]
    missed = {a: float(harm[a]["missed_harm"]) for a in _ORDER}  # type: ignore[index,call-overload]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: real approval burden ---------------------------------------------------------------
    bars = {
        "allow_all": (0.0, 0.0),
        "permission_denylist": (deny_onsurface, deny_fatigue),
        "oracle_coverage": (cov_prompt, 0.0),
    }
    x = range(len(_ORDER))
    onsurf = [bars[a][0] for a in _ORDER]
    fatigue = [bars[a][1] for a in _ORDER]
    ax_l.bar(x, onsurf, color="#2a7", label="on-surface prompt (audit budget)")
    ax_l.bar(x, fatigue, bottom=onsurf, color="#e80", label="off-surface prompt (approval fatigue)")
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels([_SHORT[a] for a in _ORDER], fontsize=8)
    ax_l.set_ylabel(f"fraction of {n:,} real tool calls prompted")
    ax_l.set_ylim(0, max(deny_prompt, cov_prompt) * 1.4 + 0.01)
    ax_l.set_title("Real approval burden: coverage spends a sparse on-surface budget")
    ax_l.legend(fontsize=8, loc="upper right")
    for i, a in enumerate(_ORDER):
        tot = bars[a][0] + bars[a][1]
        ax_l.text(i, tot + 0.002, f"{tot:.3f}", ha="center", fontsize=9)
    ax_l.text(2.0, cov_prompt + max(deny_prompt, cov_prompt) * 0.12,
              f"{(1 - cov_prompt) * 100:.1f}%\nauto-approved",
              ha="center", fontsize=8, color="#2a7")

    # --- right: harm coverage on the labeled arsenal ----------------------------------------------
    colors = {"allow_all": "#999", "permission_denylist": "#e80", "oracle_coverage": "#2a7"}
    vals = [missed[a] for a in _ORDER]
    ax_r.bar(x, vals, color=[colors[a] for a in _ORDER])
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels([_SHORT[a] for a in _ORDER], fontsize=8)
    ax_r.set_ylabel("missed-harm (labeled arsenal)")
    ax_r.set_ylim(0, 1.15)
    ax_r.set_title("Harm coverage: only the effect-surface catches every class")
    for i, v in enumerate(vals):
        ax_r.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    fig.suptitle(
        "RA21: oracle coverage gate replayed on a real Claude Code corpus "
        f"({n:,} state-mutating tool calls)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:  # pragma: no cover - local entrypoint
    root = Path(__file__).resolve().parent.parent
    summary = json.loads((root / "bench" / "cc_corpus" / "summary.json").read_text())
    out = plot_ra21(summary, root / "figures" / "ra21_cc_corpus.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
