"""FL2 composition-ablation tests (SPEC-19 §4, H70).

The contract: the four method-combinations run at one equal budget on a (smoke) flagship, every cell
is well-formed, and the H70 verdict fields are computed. The smoke model is trivial; the committed
ablation comes from the local run on the real `l@9.6k` checkpoint -- CI guarantees the apparatus is
correct and deterministic, not what the verdict is.
"""

from __future__ import annotations

import pytest

from verisim.experiments.flagship_ablation import CELLS, _policy_for, compose_verdict
from verisim.loop.policy import ConsultationPolicy


def test_policy_for_covers_every_cell():
    for cell in CELLS:
        pol = _policy_for(cell, tau=0.5, window=4, rho=0.2)
        assert isinstance(pol, ConsultationPolicy)
    with pytest.raises(ValueError):
        _policy_for("bogus", 0.5, 4, 0.2)


def test_compose_verdict_logic():
    from verisim.experiments.flagship_ablation import AblationCell

    def cell(name, h):
        return AblationCell(name, 0.2, 4, h, h, h, h / 4, 2)

    # both above the best single -> super-additive, composes, no interference
    cells = [cell("neither", 1), cell("conformal", 3), cell("speculative", 2), cell("both", 5)]
    v = compose_verdict(cells)
    assert v["h70_composes"] and v["super_additive"] and not v["interferes"]
    # both below the best single -> interferes
    cells = [cell("neither", 1), cell("conformal", 6), cell("speculative", 2), cell("both", 4)]
    v = compose_verdict(cells)
    assert v["interferes"] and not v["h70_composes"]


# --- torch-gated: the real-model ablation ---------------------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.flagship import FlagshipConfig, train_flagship  # noqa: E402
from verisim.experiments.flagship_ablation import run_ablation  # noqa: E402
from verisim.experiments.flagship_curve import FlagshipCurveConfig  # noqa: E402
from verisim.netoracle import ReferenceNetworkOracle  # noqa: E402


def test_run_ablation_well_formed():
    wm, _ = train_flagship(FlagshipConfig.smoke())
    cells = run_ablation(wm, FlagshipCurveConfig.smoke(), rho=0.3, oracle=ReferenceNetworkOracle())
    assert {c.cell for c in cells} == set(CELLS)
    for c in cells:
        assert c.h_mean >= 0.0
        assert c.budget >= 0
        assert c.ci_lo <= c.h_mean <= c.ci_hi or c.n == 1


def test_run_ablation_is_deterministic():
    wm, _ = train_flagship(FlagshipConfig.smoke())
    cfg = FlagshipCurveConfig.smoke()
    a = run_ablation(wm, cfg, rho=0.3, oracle=ReferenceNetworkOracle())
    b = run_ablation(wm, cfg, rho=0.3, oracle=ReferenceNetworkOracle())
    assert [(c.cell, c.h_mean) for c in a] == [(c.cell, c.h_mean) for c in b]
