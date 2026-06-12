"""The verisim agent-in-the-loop architecture diagram (SPEC-22 §3 — the deployment picture).

A standalone system diagram (no data) of how a computer-use agent or autonomous cyber defender uses
the verified world model as a safety layer: the propose -> preview -> verify -> allow/abort loop,
each box annotated by the research backing it. Rendered to ``figures/cu_architecture.png`` for the
README ("from foundation to application"). Run: ``python -m figures.plot_cu_architecture``.
"""

from __future__ import annotations

from pathlib import Path


def plot_architecture(path: str | Path = "figures/cu_architecture.png") -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color, tc="#111"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
            linewidth=1.6, edgecolor=color, facecolor=color + "18",
        ))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h - 8.2, sub, ha="center", va="top", fontsize=7.8, color="#333")

    def arrow(x1, y1, x2, y2, label="", color="#444", lx=0, ly=0, style="-|>"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
            linewidth=1.6, color=color, shrinkA=2, shrinkB=2,
        ))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="center",
                    fontsize=7.8, color=color, fontweight="bold")

    # row 1: the agent
    box(3, 74, 30, 20, "1 · Agent (LLM)",
        "natural-language intent →\na host action plan\n(open/write/fork/setuid/…)", "#7b3fa0")
    # row 2 left: the world model (cheap preview)
    box(3, 40, 30, 22, "2 · World model  M_θ  (cheap)",
        "imagine() — roll the plan\nforward with NO oracle:\na 'look before you leap' preview\n"
        "[SPEC-6/10: learned, 110k]", "#1f77b4")
    # row 2 right: the oracle (verifier)
    box(67, 40, 30, 22, "3 · Oracle  (free, exact)",
        "verify the preview at rate ρ —\nre-anchor to truth\n(reference oracle OR a real /bin/sh)\n"
        "[SPEC.md §2 · SPEC-11: real kernel]", "#2ca02c")
    # center: the propose-verify-correct loop
    box(38, 40, 24, 22, "propose · verify · correct",
        "M_θ drafts k steps, the oracle\nverifies & corrects on a ρ budget\n"
        "[SPEC-19 knee: ρ≈0.2 buys it]", "#ff7f0e")
    # row 3: the guardrail / safety gate
    box(20, 12, 60, 18, "4 · Safety gate — guardrail on the predicted final state",
        "content guardrail (/passwd not overwritten — credential tampering)  ·  "
        "structure guardrail (a protected daemon stays alive)\n"
        "SAFE → ALLOW (execute on the real computer)      UNSAFE → ABORT / flag\n"
        "[SPEC-20 boundary: verification is load-bearing on the CONTENT the model drifts on]",
        "#d62728")

    # arrows
    arrow(18, 74, 18, 62, "proposes plan", "#7b3fa0", lx=-8)
    arrow(33, 51, 38, 51, "draft", "#1f77b4", ly=2.5)
    arrow(62, 52, 67, 52, "", "#2ca02c")
    ax.text(64.5, 56.5, "verify ρ", ha="center", va="center", fontsize=7.5,
            color="#2ca02c", fontweight="bold")
    arrow(67, 46, 62, 46, "", "#2ca02c")
    ax.text(64.5, 42.5, "correct", ha="center", va="center", fontsize=7.5,
            color="#2ca02c", fontweight="bold")
    arrow(50, 40, 50, 30, "predicted final state", "#ff7f0e", lx=14)
    arrow(50, 12, 50, 4, "", "#d62728")
    ax.text(50, 2.4, "ALLOW → real computer        ABORT → safe", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#d62728")
    # feedback: the oracle's truth re-grounds the agent for the next step
    arrow(82, 51, 82, 84, "", "#2ca02c")
    arrow(82, 84, 33, 84, "oracle truth re-grounds the next action", "#2ca02c", ly=2.2)

    fig.suptitle(
        "Verisim — the agent-in-the-loop: a verified world model as a safety layer for "
        "computer-use agents and autonomous cyber defense",
        fontsize=12.5, fontweight="bold", y=0.99,
    )
    ax.text(50, 96.5,
            "The model is cheap but drifts; the oracle is exact but costs a call. The loop spends "
            "a small ρ budget to keep the preview faithful, so the agent's gate can be trusted.",
            ha="center", va="center", fontsize=8.5, color="#444", style="italic")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":  # pragma: no cover
    print(f"wrote {plot_architecture()}")
