"""Plot the RA4 figure (SPEC-22 H136): the command-agnostic gate, and its honest open edge.

Two panels from a :class:`~verisim.realagent.command_agnostic.RA4Result`:

  - **left -- grammar-blindness.** Of the diverse attack syntaxes (redirects, `tee`, `dd`, `python
    -c`, `sed -i`, ...), how many a grammar-specific gate can even *parse* (and therefore act on) vs
    how many the command-agnostic target catches. The grammar gate is structurally blind to most;
    the command-agnostic target catches all explicit-reference attacks.
  - **right -- the three operating rates.** Attack-catch (command-agnostic, ~1.0), benign
    false-block (~0.0 -- silent on real commands it never saw), and evasion-miss (the open edge: a
    syntactic pre-commit scan is defeated by indirection, so the principled gate routes irreversible
    dangers through CU27's exact post-commit fs-diff, which no evasion escapes).
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.command_agnostic import RA4Result


def plot_ra4(result: RA4Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: grammar-blindness (counts of attack syntaxes the gate can act on) ------------------
    n = result.n_attacks
    grammar_seen = round(result.grammar_visible * n)
    agnostic_caught = round(result.attack_target_fires * n)
    bars = ["grammar gate\n(can parse)", "command-agnostic\ntarget (catches)"]
    vals = [grammar_seen, agnostic_caught]
    colors = ["#c44", "#2a7"]
    ax_l.bar(bars, vals, color=colors, width=0.6)
    ax_l.axhline(n, color="#888", linestyle="--", linewidth=1)
    ax_l.text(1.4, n + 0.1, f"all {n} attacks", color="#888", fontsize=8, ha="right")
    ax_l.set_ylabel("attack syntaxes the gate can act on")
    ax_l.set_ylim(0, n + 1.2)
    ax_l.set_title("Grammar-blindness: the gate must not need the grammar")
    for i, val in enumerate(vals):
        ax_l.text(i, val + 0.1, str(val), ha="center", fontsize=10)

    # --- right: the three operating rates ---------------------------------------------------------
    labels = ["attack-catch\n(cmd-agnostic)", "benign\nfalse-block", "evasion-miss\n(open edge)"]
    rates = [result.attack_target_fires, result.benign_false_block, result.evasion_miss]
    rcolors = ["#2a7", "#2a7", "#c44"]
    ax_r.bar(labels, rates, color=rcolors, width=0.6)
    ax_r.set_ylabel("rate")
    ax_r.set_ylim(0, 1.18)
    ax_r.set_title("Strong where explicit; honest where evaded")
    for i, r in enumerate(rates):
        ax_r.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=10)
    ax_r.annotate("→ route to CU27\npost-commit diff (exact)", (2, result.evasion_miss),
                  textcoords="offset points", xytext=(-4, -38), fontsize=8, color="#a22",
                  ha="center")

    fig.suptitle(
        "RA4 / H136 -- the command-agnostic gate: catches attack syntaxes a grammar gate is blind "
        "to, silent on unseen benign commands; pre-commit evaded by indirection (the RA5 edge)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
