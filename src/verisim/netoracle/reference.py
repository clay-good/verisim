"""Tier-A reference network oracle (SPEC-5 §5.1): the deterministic executable truth.

A from-scratch, dependency-free model of the pinned SPEC-5 v0 semantics
(``docs/network-semantics.md``): topology/service/firewall config ops, reachability-gated
``connect``, and a virtual clock whose ``advance`` re-validates flows -- the minimal
*asynchronous* dynamic (SPEC-5 §11, W5: a link that went down earlier silently breaks a
flow, observed only at the next advance). Deterministic and pure: ``step(s, a)`` depends only
on ``(s, a)``, never mutates ``s``, and derives ``next_state`` via the shared ``apply`` so the
NW1 invariant holds by construction.

Exit-code convention: ``0`` = the action was valid (whether or not it changed state); ``1`` =
a well-defined runtime failure (connect to an unreachable service; close a non-existent flow);
``2`` = invalid arguments (unknown host, self-link).
"""

from __future__ import annotations

from verisim.net.action import NetAction
from verisim.net.state import NetworkState, can_reach, link_key
from verisim.netdelta.apply import apply
from verisim.netdelta.edits import (
    ClockAdvance,
    FlowClose,
    FlowOpen,
    FwAllow,
    FwDeny,
    HostDown,
    HostUp,
    LinkAdd,
    LinkDel,
    NetDelta,
    NetEdit,
    SetResult,
    SvcDown,
    SvcUp,
)

from .base import NetStepResult


class ReferenceNetworkOracle:
    """Deterministic interpreter of the SPEC-5 v0 network semantics."""

    version = "net-ref-1"

    def step(self, state: NetworkState, action: NetAction) -> NetStepResult:
        edits, exit_code = self._semantics(state, action)
        delta: NetDelta = [*edits, SetResult(exit_code)]
        return NetStepResult(state=apply(state, delta), delta=delta, exit_code=exit_code)

    def _semantics(self, state: NetworkState, action: NetAction) -> tuple[list[NetEdit], int]:
        name = action.name
        hosts = state.hosts

        if name == "advance":
            # Tick the clock and drop any flow that is no longer reachable (delayed effect).
            dropped = [
                FlowClose(s, d, p)
                for (s, d, p) in sorted(state.flows)
                if not can_reach(state, s, d, p)
            ]
            return [ClockAdvance(1), *dropped], 0

        if name in {"host_up", "host_down"}:
            (host,) = action.args
            if host not in hosts:
                return [], 2
            up = name == "host_up"
            if hosts[host].up == up:
                return [], 0  # idempotent no-op
            return [HostUp(host) if up else HostDown(host)], 0

        if name in {"link_up", "link_down"}:
            a, b = action.args
            if a not in hosts or b not in hosts or a == b:
                return [], 2
            present = link_key(a, b) in state.links
            if name == "link_up":
                return ([] if present else [LinkAdd(a, b)]), 0
            return ([LinkDel(a, b)] if present else []), 0

        if name in {"svc_up", "svc_down"}:
            host = action.args[0]
            port = action.port
            if host not in hosts:
                return [], 2
            listening = port in hosts[host].services
            if name == "svc_up":
                return ([] if listening else [SvcUp(host, port)]), 0
            return ([SvcDown(host, port)] if listening else []), 0

        if name in {"fw_deny", "fw_allow"}:
            host, src = action.args
            if host not in hosts or src not in hosts:
                return [], 2
            blocked = src in hosts[host].fw_deny
            if name == "fw_deny":
                return ([] if blocked else [FwDeny(host, src)]), 0
            return ([FwAllow(host, src)] if blocked else []), 0

        if name == "connect":
            src, dst = action.args[0], action.args[1]
            port = action.port
            if src not in hosts or dst not in hosts:
                return [], 2
            if not can_reach(state, src, dst, port):
                return [], 1  # connection refused / unreachable
            if (src, dst, port) in state.flows:
                return [], 0  # already established
            return [FlowOpen(src, dst, port)], 0

        if name == "close":
            src, dst = action.args[0], action.args[1]
            port = action.port
            if (src, dst, port) in state.flows:
                return [FlowClose(src, dst, port)], 0
            return [], 1  # no such flow

        raise ValueError(f"unhandled action {name!r}")  # pragma: no cover - parser guards this
