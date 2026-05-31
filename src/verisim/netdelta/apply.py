"""``apply(state, delta) -> state'`` for the network world (SPEC-5 §4, NW1).

A pure function shared by the oracle and the model's prediction path, so the NW1 invariant
``apply(state, oracle.delta) == oracle.next_state`` holds by construction. Never mutates the
input state; returns a fresh :class:`NetworkState`.
"""

from __future__ import annotations

from collections.abc import Sequence

from verisim.net.state import HostState, NetworkState, link_key

from .edits import (
    ClockAdvance,
    FlowClose,
    FlowOpen,
    FwAllow,
    FwDeny,
    HostDown,
    HostUp,
    LinkAdd,
    LinkDel,
    NetEdit,
    SetResult,
    SvcDown,
    SvcUp,
)


def _ensure_host(state: NetworkState, host: str) -> None:
    if host not in state.hosts:
        state.hosts[host] = HostState()


def apply_edit(state: NetworkState, edit: NetEdit) -> None:
    """Apply one edit in place to ``state`` (callers pass a fresh copy)."""
    if isinstance(edit, HostUp):
        _ensure_host(state, edit.host)
        state.hosts[edit.host] = state.hosts[edit.host].with_up(True)
    elif isinstance(edit, HostDown):
        _ensure_host(state, edit.host)
        state.hosts[edit.host] = state.hosts[edit.host].with_up(False)
    elif isinstance(edit, LinkAdd):
        state.links.add(link_key(edit.a, edit.b))
    elif isinstance(edit, LinkDel):
        state.links.discard(link_key(edit.a, edit.b))
    elif isinstance(edit, SvcUp):
        _ensure_host(state, edit.host)
        state.hosts[edit.host] = state.hosts[edit.host].with_service(edit.port, True)
    elif isinstance(edit, SvcDown):
        _ensure_host(state, edit.host)
        state.hosts[edit.host] = state.hosts[edit.host].with_service(edit.port, False)
    elif isinstance(edit, FwDeny):
        _ensure_host(state, edit.host)
        state.hosts[edit.host] = state.hosts[edit.host].with_fw(edit.src, True)
    elif isinstance(edit, FwAllow):
        _ensure_host(state, edit.host)
        state.hosts[edit.host] = state.hosts[edit.host].with_fw(edit.src, False)
    elif isinstance(edit, FlowOpen):
        state.flows.add((edit.src, edit.dst, edit.port))
    elif isinstance(edit, FlowClose):
        state.flows.discard((edit.src, edit.dst, edit.port))
    elif isinstance(edit, ClockAdvance):
        state.clock += edit.amount
    elif isinstance(edit, SetResult):
        state.last_exit = edit.exit_code


def apply(state: NetworkState, delta: Sequence[NetEdit]) -> NetworkState:
    """Return a fresh state with every edit in ``delta`` applied, in order."""
    out = state.copy()
    for edit in delta:
        apply_edit(out, edit)
    return out
