"""Graph delta types: the structured edits a network action makes (SPEC-5 §4).

The model's prediction target and the oracle's truth representation, mirroring v0's
``Delta`` (SPEC.md §6.1: delta prediction bounds the hallucination surface). Each edit is a
small frozen dataclass; the union is closed. ``apply`` (see ``apply.py``) is compositional
and shared by the oracle, so ``apply(state, oracle.delta) == oracle.next_state`` by
construction -- the NW1 invariant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HostUp:
    host: str


@dataclass(frozen=True)
class HostDown:
    host: str


@dataclass(frozen=True)
class LinkAdd:
    a: str
    b: str


@dataclass(frozen=True)
class LinkDel:
    a: str
    b: str


@dataclass(frozen=True)
class SvcUp:
    host: str
    port: int


@dataclass(frozen=True)
class SvcDown:
    host: str
    port: int


@dataclass(frozen=True)
class FwDeny:
    host: str
    src: str


@dataclass(frozen=True)
class FwAllow:
    host: str
    src: str


@dataclass(frozen=True)
class FlowOpen:
    src: str
    dst: str
    port: int


@dataclass(frozen=True)
class FlowClose:
    src: str
    dst: str
    port: int


@dataclass(frozen=True)
class ClockAdvance:
    amount: int = 1


@dataclass(frozen=True)
class SetResult:
    exit_code: int


NetEdit = (
    HostUp
    | HostDown
    | LinkAdd
    | LinkDel
    | SvcUp
    | SvcDown
    | FwDeny
    | FwAllow
    | FlowOpen
    | FlowClose
    | ClockAdvance
    | SetResult
)
NetDelta = list[NetEdit]
