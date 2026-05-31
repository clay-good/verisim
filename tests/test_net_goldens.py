"""NW0 golden trajectories: committed scripts -> exact canonical states (SPEC-5 §5.1).

These pin the reference oracle's semantics; CI fails on any drift, exactly as v0's
``test_goldens`` pins the filesystem semantics (SPEC-2 §12). A small 3-host config keeps the
expected canonical states readable.
"""

from verisim.net import NetworkState, parse_net_action, to_canonical
from verisim.net.config import NetConfig
from verisim.netoracle import ReferenceNetworkOracle

CONFIG = NetConfig(name="golden", hosts=("h0", "h1", "h2"), ports=(80, 443))
ORACLE = ReferenceNetworkOracle()


def _final(cmds: list[str]) -> dict[str, object]:
    state = NetworkState.initial(CONFIG.hosts)
    for cmd in cmds:
        state = ORACLE.step(state, parse_net_action(cmd)).state
    return to_canonical(state)


def _host(
    up: bool = True, services: list[int] | None = None, fw: list[str] | None = None
) -> dict[str, object]:
    return {"up": up, "services": services or [], "fw_deny": fw or []}


def test_golden_connect_then_partition_drops_flow_on_advance():
    # build a path + service, connect, then tear the path down and advance twice.
    final = _final(
        [
            "link_up h0 h1", "svc_up h1 80", "connect h0 h1 80",
            "advance", "link_down h0 h1", "advance",
        ]
    )
    assert final == {
        "hosts": {"h0": _host(), "h1": _host(services=[80]), "h2": _host()},
        "links": [],
        "flows": [],  # dropped on the second advance (path gone)
        "clock": 2,
        "last_exit": 0,
    }


def test_golden_firewall_refuses_connect():
    final = _final(["link_up h0 h1", "svc_up h1 80", "fw_deny h1 h0", "connect h0 h1 80"])
    assert final == {
        "hosts": {
            "h0": _host(),
            "h1": _host(services=[80], fw=["h0"]),
            "h2": _host(),
        },
        "links": [["h0", "h1"]],
        "flows": [],  # connection refused
        "clock": 0,
        "last_exit": 1,
    }


def test_golden_multi_service_reachability():
    final = _final(
        ["link_up h0 h1", "link_up h1 h2", "svc_up h2 443", "connect h0 h2 443"]
    )
    assert final == {
        "hosts": {"h0": _host(), "h1": _host(), "h2": _host(services=[443])},
        "links": [["h0", "h1"], ["h1", "h2"]],
        "flows": [["h0", "h2", 443]],  # reachable over the two-hop path
        "clock": 0,
        "last_exit": 0,
    }
