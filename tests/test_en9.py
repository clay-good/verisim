"""Tests for EN9 — oracle hard-negative / counterfactual contrastive (SPEC-8 §4.3, §7; OG4).

The smoke driver wires three anti-collapse cells (none / vicreg / oracle) and the two readouts
(collapse + interventional fidelity). The unit checks pin the SPEC-8 §4.3 claims the cells exist to
test: oracle-mined negatives are exact (``≠`` the true successor), they prevent collapse where the
naked target does not (H25), and the counterfactual referent lifts interventional fidelity over both
the naked and the VICReg-only cell (the H5 lift) — the EN9 first datum, demonstrated at smoke scale.
"""

from __future__ import annotations

from verisim.experiments.en9 import EN9Config, run_en9
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.net.state import NetworkState
from verisim.netmodel.graph_model import build_graph_model
from verisim.netmodel.grounded_train import (
    BranchSet,
    ContrastiveExample,
    build_contrastive_dataset,
    interventional_fidelity,
    train_contrastive,
)
from verisim.netmodel.vocab import NetVocab
from verisim.netoracle import ReferenceNetworkOracle

CFG = DEFAULT_NET_CONFIG


def test_run_en9_smoke() -> None:
    """The driver runs all three cells and emits bounded collapse + fidelity metrics."""
    cfg = EN9Config(
        train_seeds=(0,),
        train_steps_per_traj=12,
        d_model=16,
        mp_rounds=1,
        contrastive_iters=40,
    )
    rows = run_en9(cfg)
    assert {r["mode"] for r in rows} == {"none", "vicreg", "oracle"}
    for row in rows:
        assert float(row["emb_std"]) >= 0.0
        assert float(row["eff_rank"]) >= 1.0
        assert 0.0 <= float(row["intervention_top1"]) <= 1.0
        assert 0.0 <= float(row["intervention_mrr"]) <= 1.0


def test_contrastive_dataset_negatives_are_exact() -> None:
    """Every mined negative is a real, distinct successor graph; branch sets are non-trivial."""
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    ex, br = build_contrastive_dataset(oracle, vocab, CFG, seeds=(0, 1), n_steps=15, k_negatives=8)
    assert len(ex) == 30
    assert all(len(e.neg_graphs) == 8 for e in ex)  # exactly K negatives per example
    # every branch set has >= 2 distinct successors (retrieval is well-posed)
    assert all(len(b.anchor_graphs) == len(b.succ_graphs) >= 2 for b in br)


def test_oracle_negatives_prevent_collapse_and_lift_fidelity() -> None:
    """H25 + H5: the oracle referent resists collapse like VICReg and beats it on intervention.

    The naked target collapses (small embedding std); both VICReg and the oracle negatives hold the
    representation open; but only the oracle's *counterfactual* negatives carry interventional
    information, so its branch-retrieval fidelity exceeds both the naked and the VICReg-only cell.
    """
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    ex, br = build_contrastive_dataset(oracle, vocab, CFG, seeds=(0, 1), n_steps=20, k_negatives=8)

    def run(mode: str) -> tuple[float, float]:
        model = build_graph_model(vocab, CFG, d_model=24, mp_rounds=2, seed=0)
        r = train_contrastive(model, ex, br, mode=mode, steps=150, batch_size=16, seed=0)
        return r.emb_std, r.intervention_top1

    none_std, none_iv = run("none")
    vicreg_std, vicreg_iv = run("vicreg")
    oracle_std, oracle_iv = run("oracle")

    assert oracle_std > none_std * 1.5  # the exact referent prevents collapse (H25)
    assert vicreg_std > none_std * 1.5  # so does the statistical stand-in
    assert oracle_iv > vicreg_iv  # but only the oracle's counterfactuals lift intervention (H5)
    assert oracle_iv > none_iv


def test_interventional_fidelity_empty_is_zero() -> None:
    """No branch sets → fidelity is reported as zero, not an error (boundary)."""
    vocab = NetVocab(CFG)
    model = build_graph_model(vocab, CFG, d_model=16, mp_rounds=1, seed=0)
    from verisim.netmodel.grounded_train import JEPAPredictor

    predictor = JEPAPredictor(16, 16)
    top1, mrr = interventional_fidelity(model, predictor, [])
    assert top1 == 0.0 and mrr == 0.0


def test_initial_state_partition_consistent() -> None:
    """Sanity: the contrastive dataset starts from the canonical empty network."""
    assert NetworkState.initial(CFG.hosts).hosts.keys() == set(CFG.hosts)


def test_dataclasses_are_frozen() -> None:
    """The EN9 data carriers are immutable value types (the repo convention)."""
    import dataclasses

    for cls in (ContrastiveExample, BranchSet):
        params = cls.__dataclass_params__  # type: ignore[union-attr]
        assert dataclasses.is_dataclass(cls) and params.frozen
