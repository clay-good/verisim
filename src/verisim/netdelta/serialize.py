"""Graph delta <-> canonical dict/JSON serialization (SPEC-5 §4, trajectory records).

Each edit serializes to a dict tagged by ``op``; round-tripping is identity (tested in NW1).
"""

from __future__ import annotations

from typing import Any

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
    NetDelta,
    NetEdit,
    SetResult,
    SvcDown,
    SvcUp,
)


def edit_to_dict(edit: NetEdit) -> dict[str, Any]:
    if isinstance(edit, HostUp):
        return {"op": "host_up", "host": edit.host}
    if isinstance(edit, HostDown):
        return {"op": "host_down", "host": edit.host}
    if isinstance(edit, LinkAdd):
        return {"op": "link_add", "a": edit.a, "b": edit.b}
    if isinstance(edit, LinkDel):
        return {"op": "link_del", "a": edit.a, "b": edit.b}
    if isinstance(edit, SvcUp):
        return {"op": "svc_up", "host": edit.host, "port": edit.port}
    if isinstance(edit, SvcDown):
        return {"op": "svc_down", "host": edit.host, "port": edit.port}
    if isinstance(edit, FwDeny):
        return {"op": "fw_deny", "host": edit.host, "src": edit.src}
    if isinstance(edit, FwAllow):
        return {"op": "fw_allow", "host": edit.host, "src": edit.src}
    if isinstance(edit, FlowOpen):
        return {"op": "flow_open", "src": edit.src, "dst": edit.dst, "port": edit.port}
    if isinstance(edit, FlowClose):
        return {"op": "flow_close", "src": edit.src, "dst": edit.dst, "port": edit.port}
    if isinstance(edit, ClockAdvance):
        return {"op": "clock_advance", "amount": edit.amount}
    return {"op": "set_result", "exit_code": edit.exit_code}


def edit_from_dict(d: dict[str, Any]) -> NetEdit:
    op = d["op"]
    if op == "host_up":
        return HostUp(d["host"])
    if op == "host_down":
        return HostDown(d["host"])
    if op == "link_add":
        return LinkAdd(d["a"], d["b"])
    if op == "link_del":
        return LinkDel(d["a"], d["b"])
    if op == "svc_up":
        return SvcUp(d["host"], int(d["port"]))
    if op == "svc_down":
        return SvcDown(d["host"], int(d["port"]))
    if op == "fw_deny":
        return FwDeny(d["host"], d["src"])
    if op == "fw_allow":
        return FwAllow(d["host"], d["src"])
    if op == "flow_open":
        return FlowOpen(d["src"], d["dst"], int(d["port"]))
    if op == "flow_close":
        return FlowClose(d["src"], d["dst"], int(d["port"]))
    if op == "clock_advance":
        return ClockAdvance(int(d["amount"]))
    if op == "set_result":
        return SetResult(int(d["exit_code"]))
    raise ValueError(f"unknown edit op {op!r}")


def delta_to_list(delta: NetDelta) -> list[dict[str, Any]]:
    return [edit_to_dict(e) for e in delta]


def delta_from_list(items: list[dict[str, Any]]) -> NetDelta:
    return [edit_from_dict(d) for d in items]
