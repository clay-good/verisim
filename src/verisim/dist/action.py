"""The distributed action grammar (SPEC-7 §3.2, DS0 increment 1).

Actions in a constrained grammar paired with the reference oracle, exactly as every prior world
pairs a grammar with its oracle. Two of SPEC-7's three families ship in increment 1 -- the client
workload and the fault/time medium (the consensus/admin family arrives with the Raft-subset):

    put <node> <key> <val>          # client: write <val> to <key> at <node> (async repl)
    get <node> <key>                # client: read <key>'s local replica at <node> (may be stale)
    cas <node> <key> <old> <new>    # client: set <key> to <new> iff <node>'s local value == <old>
    advance <dt>                    # time: +<dt> clock; deliver in-flight msgs now due & reachable
    partition <nodes> | <nodes>     # fault: split the network into groups (| separates groups)
    heal                            # fault: remove all partitions (one fully-connected group)
    crash <node>                    # fault: <node> goes down (stops delivering/applying)
    restart <node>                  # fault: <node> comes back up

The fault/time family is the source of all interesting dynamics (stale reads under partition,
convergence after heal+advance) -- the distributed analogue of SPEC-5's ``advance Δt`` and SPEC-6's
scheduler, and the ``BUGGIFY`` of deterministic simulation testing (SPEC-7 §2.1). Transactions,
``propose``/leader ops, and message-level ``drop``/``delay``/``reorder`` are later increments.
"""

from __future__ import annotations

from dataclasses import dataclass

# name -> fixed arity (number of whitespace tokens after the name), or None for variable arity
# (``partition`` carries a ``|``-separated group list; ``heal`` takes none).
_ARITY: dict[str, int | None] = {
    "put": 3,  # node key val
    "get": 2,  # node key
    "cas": 4,  # node key old new
    "advance": 1,  # dt
    "partition": None,  # <nodes> | <nodes> [| ...]
    "heal": 0,
    "crash": 1,  # node
    "restart": 1,  # node
    # Transaction family (DS0 increment 2): a multi-key OCC transaction at a coordinator node.
    "begin": 2,  # node txn  -- open a transaction
    "tget": 3,  # node txn key  -- read <key> within the txn (pins the read version for validation)
    "tput": 4,  # node txn key val  -- buffer a write to <key> within the txn
    "commit": 2,  # node txn  -- validate the read-set + apply buffered writes atomically, or abort
    "abort": 2,  # node txn  -- discard the txn
}

# ``begin``/``commit``/``abort`` + the txn-scoped ``tget``/``tput`` are client ops (the workload).
CLIENT_OPS = frozenset({"put", "get", "cas", "begin", "tget", "tput", "commit", "abort"})
TXN_OPS = frozenset({"begin", "tget", "tput", "commit", "abort"})
FAULT_OPS = frozenset({"advance", "partition", "heal", "crash", "restart"})


class DistParseError(ValueError):
    """Raised when a string is not a valid distributed action in the DS0 grammar."""


@dataclass(frozen=True)
class DistAction:
    """A parsed distributed action. ``args`` are the post-name tokens; ``groups`` holds the parsed
    ``partition`` node groups (empty for every other op)."""

    raw: str
    name: str
    args: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...] = ()


def parse_dist_action(raw: str) -> DistAction:
    """Parse an action string into a :class:`DistAction`, validating name + arity + structure."""
    parts = raw.split()
    if not parts:
        raise DistParseError("empty action")
    name = parts[0]
    if name not in _ARITY:
        raise DistParseError(f"unknown action {name!r}; choose from {sorted(_ARITY)}")
    rest = parts[1:]

    if name == "partition":
        groups = _parse_partition_groups(rest, raw)
        return DistAction(raw=raw, name=name, args=tuple(rest), groups=groups)

    arity = _ARITY[name]
    assert arity is not None
    if len(rest) != arity:
        raise DistParseError(f"{name} expects {arity} args, got {len(rest)}: {raw!r}")
    if name == "advance":
        _parse_positive_int(rest[0], raw)
    return DistAction(raw=raw, name=name, args=tuple(rest))


def _parse_partition_groups(rest: list[str], raw: str) -> tuple[tuple[str, ...], ...]:
    """Parse ``n0 n1 | n2`` into ``(("n0","n1"), ("n2",))``; require >= 2 non-empty groups."""
    groups: list[tuple[str, ...]] = []
    current: list[str] = []
    for tok in rest:
        if tok == "|":
            if not current:
                raise DistParseError(f"empty partition group in {raw!r}")
            groups.append(tuple(current))
            current = []
        else:
            current.append(tok)
    if current:
        groups.append(tuple(current))
    if len(groups) < 2:
        raise DistParseError(f"partition needs >= 2 groups separated by '|': {raw!r}")
    seen: set[str] = set()
    for group in groups:
        for node in group:
            if node in seen:
                raise DistParseError(f"node {node!r} in more than one partition group: {raw!r}")
            seen.add(node)
    return tuple(groups)


def _parse_positive_int(tok: str, raw: str) -> int:
    try:
        value = int(tok)
    except ValueError as exc:
        raise DistParseError(f"expected an integer, got {tok!r} in {raw!r}") from exc
    if value <= 0:
        raise DistParseError(f"advance dt must be positive, got {value} in {raw!r}")
    return value
