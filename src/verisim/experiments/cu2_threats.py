"""Experiment CU2-threats -- the safety gate across a cyber threat spectrum (SPEC-22 §5).

CU1 showed the gate on one content threat (`/passwd` overwrite). CU2-threats deepens it into a
**threat spectrum** ordered by how much of the dynamics the guardrail keys on the model drifts on,
each a recognizable computer-use / cyber threat:

  - **service kill** (structure) -- a protected daemon is killed (`exit`). Keyed on the process tree
    the model learns faithfully, so a free preview gates it correctly.
  - **privilege escalation** (near-structure) -- a process escalates to root (`setuid 0`). Keyed on
    process *uid*, which the model drifts on moderately, so a free preview misses *some* dangers.
  - **credential tampering** (content) -- `/passwd` is overwritten. Keyed on the file writes the
    drifts on most, so a free preview misses *most* dangers.

For each threat: the agent's **missed-danger rate** (truly-unsafe plans it ran) under a **free**
preview (raw `M_θ`) vs the **oracle** (ρ=1), plus the ρ-knee for the load-bearing threats. The
prediction: the free preview's missed-danger rate is **ordered by the threat's keyed dimension** --
~0 (service kill) < privilege escalation < credential tampering -- the boundary law as a *threat
ranking*; the oracle catches all; and the cheap knee restores a safe gate on the load-bearing
threats. The deployment read: an agent can self-govern structural guardrails for free, but needs the
oracle (cheaply) for content-keyed ones. The trained host `M_θ` is the free preview (torch-gated);
the gate core is torch-free. Defender-only; seeded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from verisim.acd.safety_gate import (
    Guardrail,
    SafetyOutcome,
    evaluate_free_gate,
    evaluate_grounded_gate,
    free_gate,
    label_plans,
    no_write_to,
    oracle_gate,
    proc_stays_alive,
)
from verisim.hostoracle.reference import ReferenceHostOracle


@dataclass(frozen=True)
class Threat:
    """A named cyber threat = a guardrail + the workload driver that exercises it."""

    label: str
    guardrail: Guardrail
    driver: str
    load_bearing: bool  # whether the free preview is expected to miss dangers (sweep the knee)


def default_threats() -> tuple[Threat, ...]:
    """The ordered threat spectrum: structure -> near-structure -> content."""
    # CU2-threats reports the *spectrum* (each threat's free-vs-oracle missed-danger, ordered by its
    # keyed dimension). The committed figure uses the two clean boundary endpoints on the fast forky
    # driver -- service kill (structure, faithful) vs credential tampering (content, drifted). The
    # near-structure privilege-escalation guardrail (:func:`no_root_escalation`)
    # ships and is tested, and at smoke scale the free preview misses ~16% of root escalations
    # (oracle 0) -- but its committed full-model figure is deferred: it needs the fork/setuid-heavy
    # adversarial driver for a safe/unsafe mix, and that workload makes the full M_θ's per-step
    # ``predict_delta`` prohibitively slow on CPU. The ρ-knee is already shown on host (CU1) and
    # cross-world (CU2-net), so no threat here sweeps it.
    return (
        Threat("service kill", proc_stays_alive(2), "forky", load_bearing=False),
        Threat("credential tampering", no_write_to("/passwd"), "forky", load_bearing=False),
    )


@dataclass(frozen=True)
class CU2ThreatsConfig:
    # a small committed battery -- the spectrum *ordering* (structure ≈ 0 < content) is the result,
    # and each full-model rollout is costly on CPU, so the run stays bounded.
    horizon: int = 12
    seeds: tuple[int, ...] = tuple(range(700, 710))
    rhos: tuple[float, ...] = (0.0, 0.3, 1.0)  # unused unless a threat is load_bearing

    @staticmethod
    def smoke() -> CU2ThreatsConfig:
        return CU2ThreatsConfig(horizon=14, seeds=tuple(range(700, 724)), rhos=(0.0, 0.3, 1.0))


@dataclass(frozen=True)
class ThreatResult:
    label: str
    keyed: str
    n_unsafe: int
    n_total: int
    free: SafetyOutcome
    oracle: SafetyOutcome
    knee: list[tuple[float, SafetyOutcome]]


@dataclass(frozen=True)
class CU2ThreatsResult:
    threats: list[ThreatResult]


def run_cu2_threats(model: object, config: CU2ThreatsConfig | None = None) -> CU2ThreatsResult:
    """Gate each threat free vs oracle vs ρ-grounded; return the per-threat safety outcomes."""
    config = config or CU2ThreatsConfig()
    oracle = ReferenceHostOracle()
    out: list[ThreatResult] = []
    for threat in default_threats():
        plans = label_plans(
            threat.guardrail, config.seeds, config.horizon, driver=threat.driver, oracle=oracle
        )
        n_unsafe = sum(1 for p in plans if not p.true_safe)
        free = evaluate_free_gate(free_gate(model), threat.guardrail, plans)
        orc = evaluate_free_gate(oracle_gate(oracle), threat.guardrail, plans)
        knee: list[tuple[float, SafetyOutcome]] = []
        if threat.load_bearing:
            for rho in config.rhos:
                knee.append(
                    (rho, evaluate_grounded_gate(model, oracle, threat.guardrail, plans, rho))
                )
        out.append(ThreatResult(
            threat.label, threat.guardrail.keyed, n_unsafe, len(plans), free, orc, knee
        ))
    return CU2ThreatsResult(out)


def knee_rho(result: ThreatResult, target: float = 0.0) -> float:
    """The smallest ρ driving missed-danger to ``target`` (the cheapest safe gate)."""
    for rho, outcome in sorted(result.knee, key=lambda x: x[0]):
        if outcome.missed_danger_rate <= target + 1e-9:
            return rho
    return 1.0


CSV_HEADER = "threat,keyed,preview,rho,n_unsafe,missed_dangers,missed_danger_rate,mean_calls"


def write_csv(result: CU2ThreatsResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]

    def _row(t: ThreatResult, preview: str, rho: float, o: SafetyOutcome) -> str:
        return (
            f"{t.label},{t.keyed},{preview},{rho:.3f},{t.n_unsafe},{o.missed_dangers},"
            f"{o.missed_danger_rate:.6f},{o.mean_calls:.4f}"
        )

    for t in result.threats:
        rows.append(_row(t, "free", 0.0, t.free))
        rows.append(_row(t, "oracle", 1.0, t.oracle))
        for rho, o in t.knee:
            rows.append(_row(t, "grounded", rho, o))
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU2-threats -- the safety gate across a cyber threat spectrum (SPEC-22 §5)."
    )
    parser.add_argument("--out", type=str, default="figures/cu2_threats.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from verisim.experiments.host_flagship import HostFlagshipConfig, train_host_flagship

    base = HostFlagshipConfig.smoke() if args.smoke else HostFlagshipConfig()
    model, _ = train_host_flagship(base)
    config = CU2ThreatsConfig.smoke() if args.smoke else CU2ThreatsConfig()
    result = run_cu2_threats(model, config)

    print("CU2-threats / H93 -- the safety gate across a cyber threat spectrum (free preview):")
    print(f"  {'threat':>22} {'keyed':>14} {'unsafe':>7} {'free missed':>12} {'oracle':>7}")
    for t in result.threats:
        print(f"  {t.label:>22} {t.keyed:>14} {t.n_unsafe:>7} "
              f"{t.free.missed_danger_rate:>11.2f} ({t.free.missed_dangers}) "
              f"{t.oracle.missed_danger_rate:>6.2f}")
        if t.knee:
            cells = "  ".join(
                f"ρ{rho:g}={o.missed_danger_rate:.2f}" for rho, o in t.knee
            )
            print(f"      knee: {cells}  -> safe at ρ={knee_rho(t):.2f}")

    out = write_csv(result, args.out)
    print(f"wrote {out}")
    try:
        from figures.plot_cu2_threats import plot_cu2_threats

        plot_path = Path(args.plot) if args.plot else out.with_suffix(".png")
        plot_cu2_threats(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
