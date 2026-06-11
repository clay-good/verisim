"""UA6 task-taxonomy tests (SPEC-20 §7, H78).

The contract is the structural feature, the cut-budget enforcement, and the contrast logic:

  - `marginal_cut` measures how many hosts an isolation protects (0 for a leaf/compromised host,
    larger for a cut vertex);
  - the env enforces `cut_budget` — isolate disappears from the legal set once spent;
  - `run_taxonomy` produces a robust + sensitive advantage with a perfect stand-in (both ~0, the
    control), and `h78_verdict` fires only when the sensitive task's advantage exceeds the robust
    one's and is positive.

The committed verdict comes from the local run with the real drifting model.
"""

from __future__ import annotations

import pytest

from verisim.acd.containment import ContainmentConfig, ContainmentEnv, DefenderAction, OracleBackend
from verisim.acd.structural import marginal_cut, structural_action_features
from verisim.net.state import NetworkState, link_key
from verisim.netoracle import ReferenceNetworkOracle


def test_marginal_cut_measures_protection():
    # chain h0-h1-h2-h3, all up, h0 compromised. Isolating h1 cuts h2,h3 off from h0 -> protects 2.
    net = NetworkState.initial(("h0", "h1", "h2", "h3"))
    net.links = {link_key("h0", "h1"), link_key("h1", "h2"), link_key("h2", "h3")}
    comp = frozenset({"h0"})
    # isolating h1 (h0's only link) removes h1, h2, h3 from the adversary's reach -> protects 3;
    # isolating the leaf h3 protects just itself -> 1. The cut vertex protects far more (the point).
    assert marginal_cut(net, comp, "h1") == 3
    assert marginal_cut(net, comp, "h3") == 1
    assert marginal_cut(net, comp, "h1") > marginal_cut(net, comp, "h3")
    assert marginal_cut(net, comp, "h0") == 0  # compromised host


def test_structural_features_extend_basic_by_one():
    net = NetworkState.initial(("h0", "h1"))
    f = structural_action_features(net, frozenset(), DefenderAction("noop"))
    assert len(f) == 8 and f[-1] == 0.0  # noop has no cut value


def test_cut_budget_removes_isolate_when_spent():
    cfg = ContainmentConfig(n_hosts=5, n_ports=3, episode_steps=8, cut_budget=1)
    env = ContainmentEnv(cfg, OracleBackend(ReferenceNetworkOracle()))
    env.reset(seed=3)
    # spend the one isolation
    host = next(h for h in env.net.hosts if env.net.hosts[h].up)
    env.step(DefenderAction("isolate", host=host))
    # now isolate is gone from the legal set; patch/noop remain
    kinds = {a.kind for a in env.legal_actions()}
    assert "isolate" not in kinds


# --- torch-free: the contrast orchestration -------------------------------------------------------


def test_run_taxonomy_orchestration_and_verdict():
    from verisim.experiments.ua_taxonomy import TaxonomyConfig, h78_verdict, run_taxonomy
    from verisim.netloop.model import NetOracleBackedModel

    model = NetOracleBackedModel(ReferenceNetworkOracle())  # perfect -> 0 advantage (control)
    results = run_taxonomy(model, TaxonomyConfig.smoke())
    assert set(results) == {"robust", "sensitive"}
    for r in results.values():
        assert 0.0 <= r["grounded"] <= 1.0 and 0.0 <= r["free"] <= 1.0
    # a perfect model makes grounded == free on both tasks -> no advantage, H78 not supported
    assert results["robust"]["advantage"] == pytest.approx(0.0)
    assert results["sensitive"]["advantage"] == pytest.approx(0.0)
    assert not h78_verdict(results)["h78_supported"]


def test_h78_verdict_fires_on_sensitive_advantage():
    from verisim.experiments.ua_taxonomy import h78_verdict

    res = {
        "robust": {"grounded": 0.42, "free": 0.42, "advantage": 0.0},
        "sensitive": {"grounded": 0.60, "free": 0.45, "advantage": 0.15},
    }
    v = h78_verdict(res)
    assert v["h78_supported"] and v["delta"] == pytest.approx(0.15)

