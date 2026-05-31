"""Network action `A`: the fixed SPEC-5 v0 command grammar (SPEC-5 §3.2).

Two families: **config/admin** ops that change topology/services/firewall, and
**traffic/time** ops (``connect``/``close``/``advance``) that exercise and age the network.
Parsing is total over the grammar and rejects anything outside it; *semantic* validity
(does the host exist? is it reachable?) is the oracle's job, not the parser's.
"""

from __future__ import annotations

from dataclasses import dataclass

_HOST1 = {"host_up", "host_down"}  # one host arg
_HOST2 = {"link_up", "link_down"}  # two host args
_HOST_PORT = {"svc_up", "svc_down"}  # host, port
_HOST_SRC = {"fw_deny", "fw_allow"}  # host, src-host
_FLOW = {"connect", "close"}  # src, dst, port


@dataclass(frozen=True)
class NetAction:
    """A parsed network command. ``raw`` is the canonical string form.

    Argument layout by ``name``:
      - host_up/host_down            : args = (host,)
      - link_up/link_down            : args = (a, b)
      - svc_up/svc_down              : args = (host, port)
      - fw_deny/fw_allow             : args = (host, src)
      - connect/close                : args = (src, dst, port)
      - advance                      : args = ()
    """

    raw: str
    name: str
    args: tuple[str, ...]

    @property
    def port(self) -> int:
        """The port argument for svc_*/connect/close (the last positional)."""
        return int(self.args[-1])


class NetParseError(ValueError):
    """Raised when a string is not a valid command in the SPEC-5 v0 grammar."""


def parse_net_action(raw: str) -> NetAction:
    """Parse a network command string into a :class:`NetAction`, or raise."""
    text = raw.strip()
    if not text:
        raise NetParseError("empty command")
    toks = text.split()
    name, rest = toks[0], toks[1:]

    if name == "advance":
        if rest:
            raise NetParseError(f"advance takes no args: {raw!r}")
        return NetAction(raw="advance", name="advance", args=())

    if name in _HOST1:
        if len(rest) != 1:
            raise NetParseError(f"{name} needs one host: {raw!r}")
        return NetAction(raw=f"{name} {rest[0]}", name=name, args=(rest[0],))

    if name in _HOST2:
        if len(rest) != 2:
            raise NetParseError(f"{name} needs two hosts: {raw!r}")
        return NetAction(raw=f"{name} {rest[0]} {rest[1]}", name=name, args=(rest[0], rest[1]))

    if name in _HOST_SRC:
        if len(rest) != 2:
            raise NetParseError(f"{name} needs <host> <src>: {raw!r}")
        return NetAction(raw=f"{name} {rest[0]} {rest[1]}", name=name, args=(rest[0], rest[1]))

    if name in _HOST_PORT:
        if len(rest) != 2:
            raise NetParseError(f"{name} needs <host> <port>: {raw!r}")
        _require_port(rest[1], raw)
        return NetAction(raw=f"{name} {rest[0]} {rest[1]}", name=name, args=(rest[0], rest[1]))

    if name in _FLOW:
        if len(rest) != 3:
            raise NetParseError(f"{name} needs <src> <dst> <port>: {raw!r}")
        _require_port(rest[2], raw)
        return NetAction(
            raw=f"{name} {rest[0]} {rest[1]} {rest[2]}", name=name, args=(rest[0], rest[1], rest[2])
        )

    raise NetParseError(f"unknown command {name!r}")


def _require_port(token: str, raw: str) -> None:
    try:
        int(token)
    except ValueError as exc:
        raise NetParseError(f"bad port {token!r} in {raw!r}") from exc
