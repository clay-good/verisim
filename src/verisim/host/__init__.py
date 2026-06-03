"""The host world (SPEC-6): a bundle of coupled subsystems (process table + fd tables + the
embedded v0 filesystem), the syscall grammar, and canonical serialization (HC0)."""

from __future__ import annotations

from .action import HostAction, HostParseError, parse_host_action
from .config import DEFAULT_HOST_CONFIG, HostConfig
from .delta import (
    CredChange,
    FdClose,
    FdOpen,
    FsDelta,
    HostDelta,
    HostEdit,
    ProcExit,
    ProcSpawn,
    SetExit,
    apply,
    delta_from_list,
    delta_to_list,
)
from .state import (
    RUNNING,
    ZOMBIE,
    FdEntry,
    HostState,
    Process,
    from_canonical_host,
    to_canonical_host,
)

__all__ = [
    "DEFAULT_HOST_CONFIG",
    "RUNNING",
    "ZOMBIE",
    "CredChange",
    "FdClose",
    "FdEntry",
    "FdOpen",
    "FsDelta",
    "HostAction",
    "HostConfig",
    "HostDelta",
    "HostEdit",
    "HostParseError",
    "HostState",
    "ProcExit",
    "ProcSpawn",
    "Process",
    "SetExit",
    "apply",
    "delta_from_list",
    "delta_to_list",
    "from_canonical_host",
    "parse_host_action",
    "to_canonical_host",
]
