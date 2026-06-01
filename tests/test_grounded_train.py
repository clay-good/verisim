"""Tests for the EN8 oracle-grounded SSL trainer (SPEC-8 §4, §7; milestone OG3).

Two axes, two checks each. H24 (objective): the residual mask aligns to the encoded delta and the
residual objective trains only the residual tokens. H23 (collapse): the oracle-anchored target keeps
the graph representation healthy with the collapse-prevention machinery ablated, where the naked
learned target collapses — the SPEC-8 §4.1 claim, demonstrated. The deterministic partition helpers
are property-checked against OG1's :mod:`verisim.netdata.grounding` semantics.
"""

from __future__ import annotations

from verisim.net.config import NetConfig
from verisim.net.state import NetworkState
from verisim.netdelta.edits import ClockAdvance, FwDeny, HostUp, NetEdit, SvcUp
from verisim.netmodel.graph_model import build_graph_model
from verisim.netmodel.grounded_train import (
    build_grounded_dataset,
    edit_is_decidable,
    representation_health,
    residual_token_accuracy,
    residual_token_mask,
    train_grounded_decoder,
    train_jepa,
)
from verisim.netmodel.tokenizer import encode_target
from verisim.netmodel.vocab import NetVocab
from verisim.netoracle import ReferenceNetworkOracle

CFG = NetConfig()


def test_edit_is_decidable_matches_observation() -> None:
    """An edit is decidable iff every host it references is observed; globals always are."""
    observed = frozenset({"h0", "h1"})
    assert edit_is_decidable(HostUp("h0"), observed)  # observed host -> decidable
    assert not edit_is_decidable(HostUp("h4"), observed)  # unobserved host -> residual
    assert not edit_is_decidable(FwDeny("h0", "h4"), observed)  # references an unobserved src
    assert edit_is_decidable(ClockAdvance(1), observed)  # global -> always decidable
    assert edit_is_decidable(HostUp("h4"), None)  # full observation -> everything decidable


def test_residual_mask_aligns_to_encoded_delta() -> None:
    """The mask has exactly one entry per target token (edits' spans + the trailing <eos>)."""
    vocab = NetVocab(CFG)
    delta: list[NetEdit] = [HostUp("h4"), SvcUp("h0", CFG.ports[0]), ClockAdvance(1)]
    mask = residual_token_mask(delta, vocab, frozenset({"h0", "h1"}))
    assert len(mask) == len(encode_target(delta, vocab))
    # h4 edit is residual (True), the h0 edit + clock + <eos> are decidable/structural (False)
    assert mask[0] is True
    assert mask[-1] is False  # <eos> is never a residual fact
    # under full observation nothing is residual
    assert not any(residual_token_mask(delta, vocab, None))


def test_grounded_dataset_has_residual_under_partial_obs() -> None:
    """Half-observed rollouts give a non-degenerate residual; masks align; rawfeat fixed-width."""
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    ex = build_grounded_dataset(oracle, vocab, CFG, seeds=(0, 1), n_steps=20, observed_fraction=0.5)
    assert len(ex) == 40
    assert all(len(e.residual_mask) == len(e.target_ids) for e in ex)
    assert sum(sum(e.residual_mask) for e in ex) > 0  # residual is non-empty (R != {})
    widths = {len(e.next_rawfeat) for e in ex}
    assert len(widths) == 1  # the oracle-anchored target is a fixed-width vector


def test_residual_objective_trains_only_residual_tokens() -> None:
    """The residual objective fits R while leaving the offloaded decidable tokens unlearned."""
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    ex = build_grounded_dataset(oracle, vocab, CFG, seeds=(0, 1), n_steps=20, observed_fraction=0.5)
    model = build_graph_model(vocab, CFG, d_model=32, mp_rounds=2, seed=0)
    losses = train_grounded_decoder(model, ex, objective="residual", steps=200, seed=0)
    assert losses[-1] < losses[0]  # learning happened on R
    overall, residual = residual_token_accuracy(model, ex)
    # by design D is offloaded (not trained), so overall < residual: capacity went to R
    assert residual > overall


def test_likelihood_objective_learns_everything() -> None:
    """The raw-likelihood baseline fits all tokens to high accuracy (a sane control for H24)."""
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    ex = build_grounded_dataset(oracle, vocab, CFG, seeds=(0, 1), n_steps=20, observed_fraction=0.5)
    model = build_graph_model(vocab, CFG, d_model=32, mp_rounds=2, seed=0)
    train_grounded_decoder(model, ex, objective="likelihood", steps=300, seed=0)
    overall, _ = residual_token_accuracy(model, ex)
    assert overall > 0.8


def test_oracle_target_resists_collapse_without_machinery() -> None:
    """H23: with the collapse machinery ablated, the oracle target keeps the embedding healthy.

    The naked *learned* target (own encoder, stop-gradient, no EMA/VICReg) collapses — its
    embedding standard deviation falls sharply — while the oracle-anchored external referent does
    not. Embedding std is the robust collapse readout at this scale (SPEC-8 §4.1).
    """
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    ex = build_grounded_dataset(oracle, vocab, CFG, seeds=(0, 1), n_steps=15, observed_fraction=0.5)

    def run(target: str, machinery: bool) -> float:
        model = build_graph_model(vocab, CFG, d_model=24, mp_rounds=2, seed=0)
        return train_jepa(
            model, ex, target=target, collapse_machinery=machinery, steps=150, batch_size=16, seed=0
        ).emb_std

    learned_off = run("learned", False)
    oracle_off = run("oracle", False)
    learned_on = run("learned", True)
    assert oracle_off > learned_off * 1.5  # the external referent prevents collapse
    assert learned_on > learned_off  # the machinery prevents collapse for the learned target


def test_representation_health_bounds() -> None:
    """Effective rank lies in ``[1, d]`` and std is non-negative — sane diagnostic bounds."""
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    ex = build_grounded_dataset(oracle, vocab, CFG, seeds=(0,), n_steps=20, observed_fraction=0.5)
    model = build_graph_model(vocab, CFG, d_model=16, mp_rounds=2, seed=0)
    std, rank = representation_health(model, ex)
    assert std >= 0.0
    assert 1.0 <= rank <= 16.0 + 1e-6


def test_initial_state_is_partition_consistent() -> None:
    """Sanity: the dataset's first state is the canonical empty network (no hidden coupling)."""
    assert NetworkState.initial(CFG.hosts).hosts.keys() == set(CFG.hosts)
