"""Driver policies over the v0 command grammar (SPEC-2 §4).

A driver is a *stochastic policy* that emits actions to roll the oracle forward
and generate trajectories. It is not a learned agent -- v0 studies the world
model, not an agent (SPEC-2 §2.2). All randomness lives here (seeded), never in
the oracle, preserving oracle purity.

Three presets (SPEC-2 §4):
  - ``uniform``     : equal weight over commands.
  - ``weighted``    : tilted toward structure-building then -mutating, producing
                      realistic long-range dependencies.
  - ``adversarial`` : biased toward destructive/cascading commands (``rm -r``,
                      ``mv``, ``rmdir``) to stress-test compounding error (§2.4).

Arguments are sampled from existing state (so most actions are valid) but target
types are not filtered, so failure cases -- the §2.2 semantics that matter for
compounding error -- arise naturally.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from verisim.env.action import Action, parse_action
from verisim.env.config import EnvConfig
from verisim.env.state import Dir, File, State

# Command "shapes" the sampler knows how to build arguments for.
_COMMANDS = (
    "mkdir",
    "touch",
    "write",
    "append",
    "cp",
    "cp_r",
    "mv",
    "rm",
    "rm_r",
    "rmdir",
    "chmod",
    "cd",
    "cat",
    "ls",
    "export",
)

_WEIGHTS: dict[str, dict[str, float]] = {
    # The "trivial" difficulty (SPEC-2.1 §4 / K0): the smallest *learnable* world -- only
    # additive structural commands (mkdir/touch/write) creating fresh **single-segment**
    # paths directly under root (depth-1, collision-free), so every action is a clean success
    # with a short delta. The K0 control trains on this and verifies the pipeline can fit a
    # deterministic transition to exact-match (gate >= 0.95; observed 1.0). cat/ls are
    # excluded (their stdout is a harder observation); multi-segment paths are excluded
    # because the K0 finding is that exact multi-token *argument copying* (reproducing a
    # deep path in the delta) is the floor's true bottleneck -- depth-1 isolates that out so
    # the control is a clean learnability proof. The full §2.4 depth/breadth dial (K3)
    # generalizes this; "trivial" is its easiest setting.
    "trivial": {
        "mkdir": 3.0, "touch": 2.0, "write": 3.0, "append": 0.0,
        "cp": 0.0, "cp_r": 0.0, "mv": 0.0, "rm": 0.0, "rm_r": 0.0,
        "rmdir": 0.0, "chmod": 0.0, "cd": 0.0, "cat": 0.0, "ls": 0.0,
        "export": 0.0,
    },
    "uniform": dict.fromkeys(_COMMANDS, 1.0),
    "weighted": {
        "mkdir": 3.0, "touch": 3.0, "write": 3.0, "append": 2.0,
        "cp": 1.5, "cp_r": 1.0, "mv": 1.5, "rm": 1.0, "rm_r": 0.5,
        "rmdir": 1.0, "chmod": 1.0, "cd": 1.0, "cat": 1.0, "ls": 1.0,
        "export": 1.0,
    },
    "adversarial": {
        "mkdir": 1.0, "touch": 1.0, "write": 1.0, "append": 1.0,
        "cp": 1.0, "cp_r": 1.5, "mv": 3.0, "rm": 2.0, "rm_r": 4.0,
        "rmdir": 3.0, "chmod": 1.0, "cd": 1.0, "cat": 0.5, "ls": 0.5,
        "export": 0.5,
    },
}

DRIVERS = tuple(_WEIGHTS)


@dataclass
class Driver:
    """A seeded stochastic policy. ``sample(state)`` returns the next action."""

    name: str
    config: EnvConfig
    rng: random.Random

    def __post_init__(self) -> None:
        if self.name not in _WEIGHTS:
            raise ValueError(f"unknown driver {self.name!r}; choose from {DRIVERS}")

    def _dirs(self, state: State) -> list[str]:
        return [p for p, n in state.fs.items() if isinstance(n, Dir)]

    def _files(self, state: State) -> list[str]:
        return [p for p, n in state.fs.items() if isinstance(n, File)]

    def _all_paths(self, state: State) -> list[str]:
        return list(state.fs)

    def _new_path(self, state: State) -> str:
        """A fresh path under some existing directory."""
        parent_dir = self.rng.choice(self._dirs(state))
        name = self.rng.choice(self.config.name_pool)
        sep = "" if parent_dir == "/" else "/"
        return f"{parent_dir}{sep}{name}"

    def _unused_path(self, state: State, dirs: list[str] | None = None) -> str:
        """A path that does **not** already exist, so the create succeeds deterministically.

        The ``trivial`` difficulty (SPEC-2.1 §4 / K0) uses this so every action is a clean,
        collision-free success -- a genuinely learnable mapping. ``dirs`` restricts the
        candidate parents (``["/"]`` gives depth-1 single-segment paths, the K0 control); with
        the closed 8-name pool a directory fills quickly, so we search the candidate parents
        (seeded order) for the first free ``(parent, name)`` and fall back to a random fresh
        path if none is free.
        """
        dirs = list(self._dirs(state) if dirs is None else dirs)
        self.rng.shuffle(dirs)
        for parent_dir in dirs:
            names = list(self.config.name_pool)
            self.rng.shuffle(names)
            sep = "" if parent_dir == "/" else "/"
            for name in names:
                candidate = f"{parent_dir}{sep}{name}"
                if candidate not in state.fs:
                    return candidate
        return self._new_path(state)

    def sample(self, state: State) -> Action:
        weights = _WEIGHTS[self.name]
        cmd = self.rng.choices(_COMMANDS, weights=[weights[c] for c in _COMMANDS])[0]
        return parse_action(self._build(cmd, state))

    def _fresh(self, state: State) -> str:
        """A fresh create target. Trivial uses unused *depth-1* paths (copy-isolating, §4);
        other drivers use a random new path that may collide or go deep (their difficulty)."""
        if self.name == "trivial":
            return self._unused_path(state, dirs=["/"])
        return self._new_path(state)

    def _build(self, cmd: str, state: State) -> str:
        rng = self.config
        if cmd == "mkdir":
            return f"mkdir {self._fresh(state)}"
        if cmd == "touch":
            return f"touch {self._fresh(state)}"
        if cmd == "write":
            return f"write {self._fresh(state)} {self.rng.choice(rng.content_tokens)}"
        if cmd == "append":
            target = self.rng.choice(self._files(state) or self._all_paths(state))
            return f"append {target} {self.rng.choice(rng.content_tokens)}"
        if cmd == "cp":
            return f"cp {self.rng.choice(self._all_paths(state))} {self._new_path(state)}"
        if cmd == "cp_r":
            return f"cp -r {self.rng.choice(self._all_paths(state))} {self._new_path(state)}"
        if cmd == "mv":
            return f"mv {self.rng.choice(self._all_paths(state))} {self._new_path(state)}"
        if cmd == "rm":
            return f"rm {self.rng.choice(self._all_paths(state))}"
        if cmd == "rm_r":
            return f"rm -r {self.rng.choice(self._all_paths(state))}"
        if cmd == "rmdir":
            return f"rmdir {self.rng.choice(self._dirs(state))}"
        if cmd == "chmod":
            mode = self.rng.choice(rng.modes)
            return f"chmod {mode:o} {self.rng.choice(self._all_paths(state))}"
        if cmd == "cd":
            return f"cd {self.rng.choice(self._dirs(state))}"
        if cmd == "cat":
            return f"cat {self.rng.choice(self._all_paths(state))}"
        if cmd == "ls":
            return f"ls {self.rng.choice(self._all_paths(state))}"
        # export
        return f"export {self.rng.choice(rng.env_keys)}={self.rng.choice(rng.content_tokens)}"
