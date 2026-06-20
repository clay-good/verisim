"""Plot the RA18 figure (SPEC-22 H150): the abstract shell-path resolver -- the open edge, split.

Two panels:

  - **left -- disposition of the indirection battery.** Stacked bars for the regex target vs the
    resolver over the adversarial evasion battery. The regex is mostly SILENT MISS (the open edge);
    the resolver splits the edge into FIRES (string-resolvable, closed pre-commit), ABSTAIN (routed
    by reversibility -- never a silent miss), and the symlink residual (CLEAR-on-string, post-
    commit's job, proven irreducible).
  - **right -- no regression, zero silent miss.** Grouped bars: attack-fire (both 1.0, no
    regression), benign false-fire (both 0), and silent-miss on string-visible realizing commands
    (regex high, resolver 0) -- the soundness gain.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.command_agnostic import command_targets_protected
from verisim.realagent.shell_resolver import abstract_targets_protected
from verisim.realagent.shell_resolver_eval import EVASION_BATTERY, RA18Result


def plot_ra18(result: RA18Result, path: str | Path) -> Path:  # pragma: no cover - local plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # per-item disposition over the battery
    regex_miss = sum(1 for e in EVASION_BATTERY if not command_targets_protected(e.command))
    regex_catch = len(EVASION_BATTERY) - regex_miss
    fires = sum(1 for e in EVASION_BATTERY if abstract_targets_protected(e.command) == "FIRES")
    abstain = sum(1 for e in EVASION_BATTERY if abstract_targets_protected(e.command) == "ABSTAIN")
    clear = sum(1 for e in EVASION_BATTERY if abstract_targets_protected(e.command) == "CLEAR")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: stacked disposition ----------------------------------------------------------------
    bars = ["regex\ntarget (RA4)", "abstract\nresolver (RA18)"]
    silent = [regex_miss, 0]
    caught = [regex_catch, fires]
    routed = [0, abstain]
    residual = [0, clear]
    x = [0, 1]
    w = 0.55
    b1 = ax_l.bar(x, caught, w, color="#2a7", label="caught pre-commit (FIRES)")
    b2 = ax_l.bar(x, routed, w, bottom=caught, color="#e80", label="ABSTAIN -> routed (CU27)")
    base2 = [caught[i] + routed[i] for i in x]
    b3 = ax_l.bar(x, residual, w, bottom=base2, color="#999",
                  label="symlink residual (CLEAR -> post-commit)")
    base3 = [base2[i] + residual[i] for i in x]
    b4 = ax_l.bar(x, silent, w, bottom=base3, color="#c44", label="SILENT MISS (open edge)")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(bars)
    ax_l.set_ylabel(f"indirection battery items (n={len(EVASION_BATTERY)})")
    ax_l.set_title("The resolver splits the open edge; the regex leaves it silent")
    ax_l.legend(fontsize=7, loc="upper center", ncol=1)
    for bars_set in (b1, b2, b3, b4):
        for rect in bars_set:
            h = rect.get_height()
            if h > 0:
                ax_l.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + h / 2, f"{int(h)}",
                          ha="center", va="center", fontsize=9, color="white")

    # --- right: no regression, zero silent miss ---------------------------------------------------
    metrics = ["attack fire\n(no regress)", "benign\nfalse-fire", "silent miss\n(str-visible)"]
    regex_vals = [result.regex_attack_fire, result.regex_benign_false_fire,
                  result.regex_named_edge_miss]
    resolver_vals = [result.resolver_attack_fire, result.resolver_benign_false_fire,
                     result.resolver_silent_miss]
    xr = range(len(metrics))
    wr = 0.38
    ax_r.bar([i - wr / 2 for i in xr], regex_vals, wr, color="#c44", label="regex target (RA4)")
    ax_r.bar([i + wr / 2 for i in xr], resolver_vals, wr, color="#2a7", label="resolver (RA18)")
    ax_r.set_xticks(list(xr))
    ax_r.set_xticklabels(metrics, fontsize=8)
    ax_r.set_ylabel("rate")
    ax_r.set_ylim(0, 1.18)
    ax_r.set_title("No attack regression, no benign over-fire, zero silent miss")
    ax_r.legend(fontsize=8, loc="upper right")
    for i in xr:
        ax_r.text(i - wr / 2, regex_vals[i] + 0.02, f"{regex_vals[i]:.2f}", ha="center", fontsize=8)
        ax_r.text(i + wr / 2, resolver_vals[i] + 0.02, f"{resolver_vals[i]:.2f}", ha="center",
                  fontsize=8)

    fig.suptitle(
        "RA18 / H150 -- the abstract shell-path resolver: the irreversible-bash open edge splits "
        "a closed string-resolvable half and a provably-irreducible state-dependent half",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
