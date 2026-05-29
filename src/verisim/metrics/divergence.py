"""Divergence ``d(s, ŝ)`` (SPEC-2 §7.1).

Primary metric: normalized symmetric difference over the canonical state's fact
set. We represent a state as a set of distinguishable facts -- one tuple per
filesystem node ``(path, type, content_hash, mode)`` plus scalar facts (cwd, each
env binding, last exit code, last stdout hash). Scalar facts are namespaced with
a ``\\x00`` prefix so they cannot collide with filesystem paths (which begin with
``/``). Then::

    d(s, ŝ) = |facts(s) △ facts(ŝ)| / (|facts(s)| + |facts(ŝ)|)

This is the §7.1 formula with the "scalar mismatches" and "scalars" folded into
the same fact set (the scalars are just additional facts), giving ``d ∈ [0, 1]``
with ``d = 0`` iff the states are identical. The denominator is always positive
(cwd/exit/stdout facts are always present).
"""

from __future__ import annotations

from verisim.env.state import File, State

Fact = tuple[object, ...]


def state_facts(state: State) -> set[Fact]:
    """The set of distinguishable facts defining a state (see module docstring)."""
    facts: set[Fact] = set()
    for path, node in state.fs.items():
        if isinstance(node, File):
            facts.add((path, "file", node.content_hash, node.mode))
        else:
            facts.add((path, "dir", "", node.mode))
    facts.add(("\x00cwd", state.cwd))
    for key, value in state.env.items():
        facts.add(("\x00env", key, value))
    facts.add(("\x00exit", state.last.exit_code))
    facts.add(("\x00stdout", state.last.stdout_hash))
    return facts


def divergence(a: State, b: State) -> float:
    """Normalized symmetric difference. ``0.0`` iff ``a`` and ``b`` are identical."""
    fa = state_facts(a)
    fb = state_facts(b)
    denom = len(fa) + len(fb)
    return len(fa ^ fb) / denom
