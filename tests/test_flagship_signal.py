"""FL6 signal-diagnostic tests (SPEC-19 §4, H77).

The contract is the ranking-vs-calibration measurement:

  - `signal_diagnostic` computes Spearman + trigger-precision lift, and the H77 flag fires when the
    signal ranks drift and beats the base rate;
  - a perfectly-ranking signal (signal == divergence) gives Spearman 1 and full trigger precision;
  - an uninformative (constant) signal gives no precision lift;
  - on a real smoke model, the collector returns aligned (signal, divergence) pairs.

The committed numbers come from the local run on the real `l@9.6k` checkpoint; CI proves the math.
"""

from __future__ import annotations

import pytest

from verisim.experiments.flagship_signal import signal_diagnostic


def test_perfect_signal_ranks_and_triggers():
    # signal == divergence: perfect ranking, and the top-budget steps are exactly the biggest divs
    pairs = [(0.1, 0.1), (0.0, 0.0), (0.9, 0.9), (0.5, 0.5), (0.3, 0.3)]
    d = signal_diagnostic(pairs, epsilon=0.2, budget_frac=0.4)
    assert d["spearman"] == pytest.approx(1.0)
    assert d["trigger_precision"] >= d["base_breach_rate"]
    assert d["h77_supported"]


def test_uninformative_signal_has_no_lift():
    # constant signal -> no ranking, trigger precision can't beat the base rate meaningfully
    pairs = [(0.5, 0.0), (0.5, 0.0), (0.5, 0.9), (0.5, 0.9)]
    d = signal_diagnostic(pairs, epsilon=0.2, budget_frac=0.5)
    assert d["precision_lift"] <= 1e-9  # no edge over base rate
    assert not d["h77_supported"]


def test_anti_correlated_signal_not_supported():
    # signal decreases as divergence increases -> negative Spearman -> H77 not supported
    pairs = [(0.9, 0.0), (0.6, 0.3), (0.3, 0.6), (0.0, 0.9)]
    d = signal_diagnostic(pairs, epsilon=0.2)
    assert d["spearman"] < 0
    assert not d["h77_supported"]


# --- torch-gated: the real-model collection -------------------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.flagship import FlagshipConfig, train_flagship  # noqa: E402
from verisim.experiments.flagship_curve import FlagshipCurveConfig  # noqa: E402
from verisim.experiments.flagship_signal import collect_signal_divergence  # noqa: E402
from verisim.net.config import DEFAULT_NET_CONFIG  # noqa: E402
from verisim.netoracle import ReferenceNetworkOracle  # noqa: E402


def test_collect_signal_divergence_aligned():
    wm, _ = train_flagship(FlagshipConfig.smoke())
    pairs = collect_signal_divergence(
        wm, FlagshipCurveConfig.smoke(), ReferenceNetworkOracle(), DEFAULT_NET_CONFIG
    )
    assert len(pairs) > 0
    for s, dvg in pairs:
        assert s >= 0.0 and dvg >= 0.0
