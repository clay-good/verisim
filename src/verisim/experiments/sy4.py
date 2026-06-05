"""SY4 -- the determinism attestation (SPEC-11 §2.4, §5; milestone SO2; hypothesis H30).

What licenses calling SY1 *true* determinism rather than recorded replay. On the
single-threaded v0 grammar the system oracle is a pure function of ``(state, action)``
under the :class:`~verisim.oracle.sandbox_seal.DeterminismSeal`: the grammar reads no
clock, no RNG, and runs no concurrency, so a sealed step must reproduce bit-for-bit across
repeats. This experiment proves it: re-run a fixed ``(s, a)`` battery ``N`` times and assert
every repeat yields a bit-identical canonical state, exit code, and stdout, and that the
``determinism_report`` accurately enumerates what is sealed.

Refuted (H30) if any repeat diverges while the report claims all-sealed -- which would mean
an *undeclared* nondeterminism source, the one genuinely adverse outcome SPEC-11 names. The
determinism-report honesty surface exists precisely to make that loud rather than silent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.env.action import Action, parse_action
from verisim.env.serialize import to_canonical_str
from verisim.env.state import State
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

# A fixed battery touching every command family -- built by rolling the reference oracle
# forward over a deterministic script, so each (state, action) is a concrete, reproducible
# transition. The script exercises create / modify / move / copy / chmod / read / nav / env.
_SCRIPT = (
    "mkdir /a", "touch /a/f", "write /a/f alpha", "append /a/f beta", "mkdir /a/b",
    "cp /a/f /a/b/g", "mv /a/b/g /a/b/h", "chmod 600 /a/f", "cat /a/f", "ls /a",
    "cd /a", "export HOME=omega", "cp -r /a /c", "rm /a/f", "rmdir /a/b",
)


def _battery() -> list[tuple[State, Action]]:
    """The fixed ``(state, action)`` pairs: each prefix-state of the deterministic script."""
    ref = ReferenceOracle()
    state = State.empty()
    pairs: list[tuple[State, Action]] = []
    for raw in _SCRIPT:
        action = parse_action(raw)
        pairs.append((state, action))
        state = ref.step(state, action).state
    return pairs


def _signature(oracle: SandboxOracle, state: State, action: Action) -> str:
    """The bit-exact signature of one step: canonical state + exit + stdout."""
    r = oracle.step(state, action)
    return f"{to_canonical_str(r.state)}|{r.exit_code}|{r.stdout!r}"


@dataclass
class SY4Result:
    available: bool
    platform: str
    n_repeats: int
    n_transitions: int
    bit_identical: bool
    seal_all_sealed: bool
    first_divergence: str  # "" if none
    report_notes: str


def run_sy4(*, n_repeats: int = 8, oracle: SandboxOracle | None = None) -> SY4Result:
    """Run the determinism battery ``n_repeats`` times; assert every repeat is bit-identical."""
    import sys

    try:
        oracle = oracle or SandboxOracle()
    except SystemOracleUnavailable:
        return SY4Result(False, sys.platform, n_repeats, 0, False, False, "no shell", "")

    pairs = _battery()
    baseline = [_signature(oracle, s, a) for s, a in pairs]
    bit_identical = True
    first_div = ""
    for _ in range(n_repeats - 1):
        repeat = [_signature(oracle, s, a) for s, a in pairs]
        for i, (b, r) in enumerate(zip(baseline, repeat, strict=True)):
            if b != r:
                bit_identical = False
                first_div = f"transition {i} ({pairs[i][1].raw!r}) diverged across repeats"
                break
        if not bit_identical:
            break

    report = oracle.determinism_report()
    all_sealed = (
        report.clock_sealed and report.rng_sealed
        and report.concurrency_sealed and report.env_leakage_sealed
    )
    return SY4Result(
        available=True,
        platform=sys.platform,
        n_repeats=n_repeats,
        n_transitions=len(pairs),
        bit_identical=bit_identical,
        seal_all_sealed=all_sealed,
        first_divergence=first_div,
        report_notes=report.notes,
    )


CSV_HEADER = "metric,value"


def write_csv(result: SY4Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        CSV_HEADER,
        f"platform,{result.platform}",
        f"available,{result.available}",
        f"n_repeats,{result.n_repeats}",
        f"n_transitions,{result.n_transitions}",
        f"bit_identical,{result.bit_identical}",
        f"seal_all_sealed,{result.seal_all_sealed}",
        f"first_divergence,{result.first_divergence or '(none)'}",
    ]
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run SY4 (system-oracle determinism attestation).")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out", type=str, default="runs/sy4/records.jsonl")
    parser.add_argument("--csv", type=str, default="figures/sy4_determinism.csv")
    args = parser.parse_args()
    result = run_sy4(n_repeats=args.repeats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {
        "platform": result.platform, "available": result.available,
        "n_repeats": result.n_repeats, "n_transitions": result.n_transitions,
        "bit_identical": result.bit_identical, "seal_all_sealed": result.seal_all_sealed,
        "first_divergence": result.first_divergence,
    }
    Path(args.out).write_text(json.dumps(rec) + "\n")
    path = write_csv(result, args.csv)
    print(f"wrote {path}  (platform={result.platform})")
    print(f"  bit_identical={result.bit_identical} over {result.n_repeats} repeats "
          f"x {result.n_transitions} transitions; all_sealed={result.seal_all_sealed}")
    if result.available and not (result.bit_identical and result.seal_all_sealed):
        raise SystemExit(f"DETERMINISM FAILURE: {result.first_divergence or 'seal not all-sealed'}")


if __name__ == "__main__":  # pragma: no cover
    main()
