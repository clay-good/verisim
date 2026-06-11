"""UA6 -- the drift-sensitive task's structural feature (SPEC-20 §7, H78 — the H74 redirect).

H74 was refuted because the UA0 containment policy keys on *local, drift-invariant* features (is a
host compromised / adjacent to a compromised one), so a model that drifts on reachability still
teaches the right policy. The redirect (SPEC-20 §7): build a task whose optimal policy
depends on the **multi-hop reachability the model actually drifts on**, testing whether faithfulness
becomes load-bearing there.

The lever is one structural feature: the **marginal cut** of an isolate action — how many
currently-uncompromised hosts stop being reachable from the compromised set if this host is brought
down. A leaf host protects ~itself; an articulation point protects many. This is a *global*,
multi-hop quantity (it depends on the whole reachability graph, not one edge), exactly what a
drifting reachability model gets wrong — and, paired with a tight isolation budget (`cut_budget`, so
*which* host you cut matters), it makes the optimal policy depend on faithful reachability. In
`E_grounded` the feature is computed on oracle-corrected state; in `E_free` on the drifted
state. H78 asks whether grounding now buys transfer where UA0 found it did not.
"""

from __future__ import annotations

from verisim.net.state import NetworkState, connected_hosts

from .containment import DefenderAction
from .policy import action_features

# basic 7 features + the marginal-cut structural feature
N_STRUCTURAL_FEATURES = 8


def reachable_from_compromised(net: NetworkState, compromised: frozenset[str]) -> set[str]:
    """The set of hosts link-reachable from any compromised host (the adversary's frontier)."""
    out: set[str] = set()
    for src in compromised:
        out |= connected_hosts(net, src)
    return out


def marginal_cut(net: NetworkState, compromised: frozenset[str], host: str) -> int:
    """How many uncompromised hosts isolating ``host`` removes from the adversary's reach.

    The multi-hop quantity the optimal limited-budget defender keys on, and exactly what a drifting
    reachability model corrupts: bring ``host`` down, recompute the compromised-reachable set, and
    count how many uncompromised hosts it protects.
    """
    hs = net.hosts.get(host)
    if host in compromised or hs is None or not hs.up:
        return 0
    before = reachable_from_compromised(net, compromised) - compromised
    net2 = net.copy()
    net2.hosts[host] = hs.with_up(False)
    after = reachable_from_compromised(net2, compromised) - compromised
    return max(0, len(before) - len(after))


def structural_action_features(
    net: NetworkState, compromised: frozenset[str], action: DefenderAction
) -> list[float]:
    """The basic drift-robust features plus the marginal-cut of an isolate action (normalized)."""
    base = action_features(net, compromised, action)
    cut = 0.0
    if action.kind == "isolate" and action.host:
        cut = marginal_cut(net, compromised, action.host) / max(1, len(net.hosts))
    return [*base, cut]
