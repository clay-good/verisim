"""Host drift-profile tests (SPEC-20 §7 host fork).

Torch-gated (needs a host model). The contract: per-action accuracy and per-dimension drift are
well-formed (accuracies in [0,1], drift rates in [0,1], counts positive). The committed numbers come
from the local run on the real host `l` checkpoint; CI proves the apparatus.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.experiments.host_drift import (  # noqa: E402
    HostDriftConfig,
    dimension_drift,
    per_action_accuracy,
    run_host_drift,
)
from verisim.experiments.host_flagship import HostFlagshipConfig, train_host_flagship  # noqa: E402


def _smoke_model():
    model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    return model


def test_per_action_accuracy_well_formed():
    acc = per_action_accuracy(_smoke_model(), HostDriftConfig.smoke())
    assert acc  # at least one verb exercised
    for d in acc.values():
        assert 0.0 <= d["accuracy"] <= 1.0
        assert d["n"] > 0


def test_dimension_drift_well_formed():
    dim = dimension_drift(_smoke_model(), HostDriftConfig.smoke())
    for key in ("proc_drift", "fd_drift", "fs_drift"):
        assert 0.0 <= dim[key] <= 1.0
    assert dim["n"] > 0


def test_run_host_drift_combines_both():
    profile = run_host_drift(_smoke_model(), HostDriftConfig.smoke())
    assert "per_action" in profile and "dimension" in profile
