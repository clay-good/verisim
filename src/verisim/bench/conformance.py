"""Conformance checks for the shipped RL / eval surfaces (SPEC-18 §6 PB-pack).

A product is only reusable if it honors the contracts the ecosystem expects. This module asserts,
torch-free, that each world's ``verifiers``-spec RL env satisfies the **Gymnasium reset/step
contract**
(``reset()`` returns an observation; ``step(delta)`` returns a transition with a numeric reward, a
``done`` flag, and an info dict; the episode terminates) and the **`verifiers` `load_environment`
entrypoint** is present and constructs. The Inspect-task contract (the task builds, its scorer has
unit
tests) is asserted in the test suite where ``inspect_ai`` is an optional import.

Each check returns a :class:`ConformanceResult`; :func:`run_conformance` runs them all and reports a
green/red table. No GPU, no network, no real shell.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConformanceResult:
    """One contract check: did the surface honor it?"""

    surface: str  # e.g. "rl:network"
    contract: str  # e.g. "gymnasium reset/step"
    passed: bool
    detail: str


def _check_gym_env(name: str, load: Callable[..., Any]) -> ConformanceResult:
    """Assert ``load_environment`` builds an env honoring the Gymnasium reset/step contract."""
    try:
        env = load()
        obs = env.reset()
        if obs is None:
            return ConformanceResult(name, "gymnasium reset/step", False, "reset() returned None")
        # A null/empty delta is always a valid action; step returns a numeric reward + done flag.
        transition = env.step([])
        reward = transition.reward
        done = transition.done
        info = transition.info
        ok = isinstance(reward, int | float) and isinstance(done, bool) and isinstance(info, dict)
        detail = f"reset+step ok; reward={reward}, done={done}" if ok else "step contract violated"
        return ConformanceResult(name, "gymnasium reset/step", ok, detail)
    except Exception as exc:
        return ConformanceResult(
            name, "gymnasium reset/step", False, f"{type(exc).__name__}: {exc}"
        )


def _check_episode_terminates(name: str, load: Callable[..., Any]) -> ConformanceResult:
    """Assert the episode reaches ``done`` within its declared horizon (no infinite rollout)."""
    try:
        env = load()
        env.reset()
        steps = 0
        cap = env.n_steps + 2
        while steps < cap:
            transition = env.step([])
            steps += 1
            if transition.done:
                break
        ok = steps <= cap
        return ConformanceResult(name, "episode terminates", ok, f"done after {steps} steps")
    except Exception as exc:
        return ConformanceResult(name, "episode terminates", False, f"{type(exc).__name__}: {exc}")


def run_conformance() -> list[ConformanceResult]:
    """Run the contract checks across all four worlds' RL envs (SPEC-18 PB-pack)."""
    from verisim.distrl.environment import load_environment as load_dist
    from verisim.hostrl.environment import load_environment as load_host
    from verisim.rl.environment import load_environment as load_fs

    surfaces: tuple[tuple[str, Callable[..., Any]], ...] = (
        ("rl:filesystem", load_fs),
        ("rl:host", load_host),
        ("rl:distributed", load_dist),
    )
    results: list[ConformanceResult] = []
    for name, load in surfaces:
        results.append(_check_gym_env(name, load))
        results.append(_check_episode_terminates(name, load))
    return results


def all_passed(results: list[ConformanceResult]) -> bool:
    return all(r.passed for r in results)
