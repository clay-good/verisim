"""The host syscall grammar (SPEC-6 §3.2, HC0 increment 1).

Actions are syscalls in a constrained grammar paired with the oracle, exactly as v0 pairs a shell
grammar with :class:`~verisim.oracle.reference.ReferenceOracle`. Every syscall names the **acting
pid** explicitly (the scheduler/"current process" is a later HC increment). The increment-1 subset:

    fork <pid>                     # process: create a child of <pid>
    exit <pid> <code>              # process: <pid> becomes a ZOMBIE with this exit code
    kill <pid> <target>            # process: <pid> terminates <target> (gated: same-uid or root)
    setuid <pid> <uid>             # creds: change <pid>'s uid (root-only; the privilege axis)
    open <pid> <path>              # files: bind a new fd for <pid> to <path> -> returns fd
    write <pid> <fd> <token>       # files: write <token> through <fd> (delegated to the FS oracle)
    read <pid> <fd>                # files: read <fd>'s content back (delegated to the FS oracle)
    close <pid> <fd>               # files: release <fd>

``fork``/``exit``/``kill`` are the long-range, branching, compounding-state process core (SPEC-6
§3.2) -- ``kill`` is the inter-process control op, **permission-gated** like a real OS (a process
may terminate another only if it is root or shares the target's uid, the EPERM rule -- the
privilege axis again); ``setuid`` makes privilege state first-class; ``open``/``write``/``read``/
``close`` exercise the per-process fd table over the embedded v0 filesystem -- ``read`` closing the
write/read round trip (a host you can write to but not read from is incomplete). Without a per-fd
offset (a later increment), ``read`` returns the file's whole content, read-only. Sockets, IPC,
``wait``, ``dup``/``lseek``, ``chdir``, the scheduler (``yield``/``advance``) are later increments.
"""

from __future__ import annotations

from dataclasses import dataclass

_ARITY: dict[str, int] = {
    "fork": 1,  # pid
    "exit": 2,  # pid code
    "kill": 2,  # pid target
    "setuid": 2,  # pid uid
    "open": 2,  # pid path
    "write": 3,  # pid fd token
    "read": 2,  # pid fd
    "close": 2,  # pid fd
}


class HostParseError(ValueError):
    """Raised when a string is not a valid host syscall in the HC0 grammar."""


@dataclass(frozen=True)
class HostAction:
    """A parsed syscall. ``pid`` is the acting process; ``args`` are the remaining tokens."""

    raw: str
    name: str
    pid: int
    args: tuple[str, ...]


def parse_host_action(raw: str) -> HostAction:
    """Parse a syscall string into a :class:`HostAction`, validating name + arity + pid (§3.2)."""
    parts = raw.split()
    if not parts:
        raise HostParseError("empty action")
    name = parts[0]
    if name not in _ARITY:
        raise HostParseError(f"unknown syscall {name!r}; choose from {sorted(_ARITY)}")
    if len(parts) != _ARITY[name] + 1:
        raise HostParseError(f"{name} takes {_ARITY[name]} args, got {len(parts) - 1}: {raw!r}")
    try:
        pid = int(parts[1])
    except ValueError as exc:
        raise HostParseError(f"{name}: pid must be an integer, got {parts[1]!r}") from exc
    return HostAction(raw=raw, name=name, pid=pid, args=tuple(parts[2:]))
