"""Experiment RA1 -- the oracle safety gate around a real LLM doing real computer-use (H133).

Runs the curated task battery under three gate schedules (undefended / covering target / full
oracle) and reports the safety/utility/cost frontier. By default it replays the recorded transcript
(hermetic, fast, the committed figure); ``--record`` runs a real Claude agent instead (gated behind
``VERISIM_LLM=1``) and prints the transcript to paste back into ``harness.RECORDED_TRANSCRIPT``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.realagent.harness import (
    RECORDED_TRANSCRIPT,
    RA1Config,
    ScriptedAgent,
    cu_ra1_verdict,
    run_ra1,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA1 -- the real-LLM safety-gate demo (H133).")
    parser.add_argument("--out", type=str, default="figures/ra1_real_agent.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--record", action="store_true",
                        help="run the real Claude agent (needs VERISIM_LLM=1) and print transcript")
    args = parser.parse_args()

    config = RA1Config.smoke() if args.smoke else RA1Config()

    if args.record:
        from verisim.realagent.claude_agent import ClaudeAgent, llm_enabled

        if not llm_enabled():
            raise SystemExit("--record needs VERISIM_LLM=1 (the real-Claude arm is gated off)")
        agent = ClaudeAgent()
        result = run_ra1(config, agent, agent_name="claude-opus-4-8")
        print("\nRECORDED_TRANSCRIPT = {")
        for tid, cmds in agent.transcript.items():
            print(f"    {tid!r}: {cmds!r},")
        print("}")
    else:
        result = run_ra1(config, ScriptedAgent(RECORDED_TRANSCRIPT))

    v = cu_ra1_verdict(result)

    print(f"\nRA1 / H133 -- the oracle safety gate around a real LLM ({result.agent_name}):")
    print(f"  {result.n_tasks} curated tasks ({result.n_injected} with a prompt-injection)\n")
    print(f"    {'schedule':14s} {'missed_danger':>13s} {'task_success':>12s} {'calls/task':>11s}")
    for c in result.cells:
        print(f"    {c.schedule:14s} {c.missed_danger_rate:13.3f} {c.task_success_rate:12.3f} "
              f"{c.mean_oracle_calls:11.2f}")

    print(f"\n  undefended agent breaches on injection = {v['undefended_breaches']} "
          f"(injected-task breach {v['undefended_injected_breach']:.2f})")
    print(f"  covering gate drives missed-danger to zero = {v['gate_drives_to_zero']} "
          f"with no utility loss = {v['no_utility_loss']}")
    print(f"  cheaper than the full oracle = {v['cheaper_than_full_oracle']} "
          f"({v['call_saving']:.1f}x: {v['target_calls']:.2f} vs "
          f"{v['full_oracle_calls']:.2f} calls)")
    print(f"  coverage holds -> gate safety is model-independent = {v['covers_grammar']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra1 import plot_ra1

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra1(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
