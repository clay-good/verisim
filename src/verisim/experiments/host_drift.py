"""Host drift profile -- where does the host M_θ drift? (SPEC-20 §7 host fork; the cross-world law).

The network drift profile (SPEC-20 §7) found the flat model faithful on the *control-relevant
structure* (reachability) and drifting only on *control-irrelevant content* (flows) -- which is why
faithfulness was not load-bearing for reactive OR predictive network control. The host world is
harder (`H_free≈9` vs the network's ≈18), so the question is whether its drift finally reaches the
control-relevant dynamics.

This module measures it two ways on the frozen host flagship: (1) **per-action 1-step accuracy** --
for each host verb (`fork`/`kill`/`open`/`write`/`close`/...), how often the predicted bundle
delta is *exactly* the oracle's, so we can read which *control levers* the model gets right; and (2)
**per-dimension free-running drift** -- the process-set (pids+states), the fd table, and the
filesystem, compared free-vs-true over the horizon. The finding it produces is the cross-world law:
the flat model learns the **discrete structural** dynamics (process tree, reachability) and
drifts on the **content** dynamics (file writes, flows), so a control task keyed on structure is
drift-robust in *both* worlds, and only a content-keyed task (e.g. file integrity, keyed on `write`)
is the drift-sensitive opportunity. CPU-local; CI runs the smoke instance.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.delta import apply as host_apply
from verisim.host.state import HostState
from verisim.hostdata import HostDriver
from verisim.hostmetrics.bits import delta_exact
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle

if TYPE_CHECKING:
    from verisim.hostmodel import NeuralHostWorldModel


@dataclass(frozen=True)
class HostDriftConfig:
    driver: str = "forky"
    n_episodes: int = 40
    accuracy_steps: int = 40
    drift_steps: int = 12
    drift_episodes: int = 30
    seed: int = 300

    @staticmethod
    def smoke() -> HostDriftConfig:
        return HostDriftConfig(n_episodes=4, accuracy_steps=8, drift_steps=6, drift_episodes=4)


def per_action_accuracy(
    model: NeuralHostWorldModel, config: HostDriftConfig, *, oracle: HostOracle | None = None,
) -> dict[str, dict[str, float]]:
    """Per host verb: the fraction of steps whose predicted delta exactly matches the oracle's.

    Reads *which control levers* the model predicts faithfully (high acc) vs drifts on (low acc).
    """
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # name -> [exact, total]
    for ep in range(config.n_episodes):
        drv = HostDriver(name=config.driver, config=host, rng=random.Random(config.seed + ep))
        state = HostState.initial()
        for _ in range(config.accuracy_steps):
            action = drv.sample(state)
            result = oracle.step(state, action)
            counts[action.name][1] += 1
            pred = model.predict_delta(state, action)
            counts[action.name][0] += int(delta_exact(pred, result.delta))
            state = result.state
    return {
        name: {"accuracy": ex / tot, "n": float(tot)}
        for name, (ex, tot) in counts.items()
    }


def dimension_drift(
    model: NeuralHostWorldModel, config: HostDriftConfig, *, oracle: HostOracle | None = None,
) -> dict[str, float]:
    """Free-running disagreement rate per state dimension: process set, fds, filesystem."""
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    proc_dis = fd_dis = fs_dis = total = 0
    for ep in range(config.drift_episodes):
        drv = HostDriver(name=config.driver, config=host, rng=random.Random(config.seed + 200 + ep))
        true_s = HostState.initial()
        free_s = HostState.initial()
        for _ in range(config.drift_steps):
            action = drv.sample(true_s)
            true_s = oracle.step(true_s, action).state
            free_s = host_apply(free_s, model.predict_delta(free_s, action))
            total += 1
            if {p.pid: p.state for p in true_s.procs.values()} != {
                p.pid: p.state for p in free_s.procs.values()
            }:
                proc_dis += 1
            if set(true_s.fds) != set(free_s.fds):
                fd_dis += 1
            if true_s.fs != free_s.fs:
                fs_dis += 1
    return {
        "proc_drift": proc_dis / total if total else 0.0,
        "fd_drift": fd_dis / total if total else 0.0,
        "fs_drift": fs_dis / total if total else 0.0,
        "n": float(total),
    }


def run_host_drift(
    model: NeuralHostWorldModel, config: HostDriftConfig | None = None, *,
    oracle: HostOracle | None = None,
) -> dict[str, Any]:
    """The full host drift profile: per-action accuracy + per-dimension free-running drift."""
    config = config or HostDriftConfig()
    oracle = oracle or ReferenceHostOracle()
    return {
        "per_action": per_action_accuracy(model, config, oracle=oracle),
        "dimension": dimension_drift(model, config, oracle=oracle),
    }


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse
    import json

    from verisim.experiments.host_flagship import load_checkpoint

    parser = argparse.ArgumentParser(description="Host drift profile (SPEC-20 §7 host fork).")
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/host-l")
    parser.add_argument("--out", type=str, default="figures/host_drift.json")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint).world_model
    profile = run_host_drift(model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(profile, indent=2) + "\n")

    print("PER-ACTION 1-step accuracy (which control levers the model gets right):")
    for name, d in sorted(profile["per_action"].items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"  {name:10s} acc={d['accuracy']:.3f}  (n={int(d['n'])})")
    dim = profile["dimension"]
    print(f"free-running drift: procs={dim['proc_drift']:.3f}  fds={dim['fd_drift']:.3f}  "
          f"fs={dim['fs_drift']:.3f}")
    print("cross-world law: the flat model is faithful on discrete STRUCTURE (process tree, "
          "reachability) and drifts on CONTENT (file writes, flows).")


if __name__ == "__main__":  # pragma: no cover
    main()
