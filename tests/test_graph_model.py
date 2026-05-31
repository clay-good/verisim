"""Tests for the GNN+RSSM graph arm of M_θ (SPEC-5 §6.1-6.2, the NW8 arm).

These prove the arm is a valid, learnable drop-in: it satisfies the loop's model protocols,
every prediction is a grammar-valid delta regardless of weights, it exposes the RSSM belief-variance
uncertainty signal, it is deterministic, and gradients flow through the whole graph→RSSM→decoder
stack (the training path increment 3 formalizes).
"""

from __future__ import annotations

import torch

from verisim.net.action import parse_net_action
from verisim.net.config import NetConfig
from verisim.net.state import HostState, NetworkState, link_key
from verisim.netdelta.edits import NetEdit
from verisim.netloop.model import NetModel, NetUncertaintyModel
from verisim.netmodel.graph import build_graph
from verisim.netmodel.graph_model import build_graph_model, graphs_to_tensors
from verisim.netmodel.tokenizer import encode_target
from verisim.netmodel.vocab import NetVocab

CFG = NetConfig()


def _state() -> NetworkState:
    return NetworkState(
        hosts={
            "h0": HostState(up=True, services=(22, 80), fw_deny=("h3",)),
            "h1": HostState(up=True, services=(), fw_deny=()),
            "h2": HostState(up=True, services=(443,), fw_deny=("h0",)),
            "h3": HostState(up=True, services=(), fw_deny=()),
            "h4": HostState(up=True, services=(22,), fw_deny=()),
        },
        links={link_key("h0", "h2"), link_key("h2", "h4")},
        flows={("h0", "h2", 443)},
        clock=3,
        last_exit=0,
    )


def test_implements_loop_protocols() -> None:
    vocab = NetVocab(CFG)
    model = build_graph_model(vocab, CFG, seed=0)
    assert isinstance(model, NetModel)
    assert isinstance(model, NetUncertaintyModel)


def test_predicts_grammar_valid_delta_untrained() -> None:
    vocab = NetVocab(CFG)
    model = build_graph_model(vocab, CFG, seed=0)
    for cmd in ["svc_up h1 80", "connect h0 h2 443", "fw_deny h2 h4", "advance", "host_down h4"]:
        delta = model.predict_delta(_state(), parse_net_action(cmd))
        assert isinstance(delta, list)
        assert all(isinstance(e, NetEdit) for e in delta)
        # valid by construction: re-encodes to a parseable target
        assert encode_target(delta, vocab)[-1] == vocab.eos


def test_uncertainty_is_nonnegative_float() -> None:
    vocab = NetVocab(CFG)
    model = build_graph_model(vocab, CFG, seed=0)
    delta, belief_var = model.predict_delta_with_uncertainty(
        _state(), parse_net_action("svc_up h1 80")
    )
    assert isinstance(belief_var, float)
    assert belief_var >= 0.0
    assert all(isinstance(e, NetEdit) for e in delta)


def test_deterministic_same_seed() -> None:
    vocab = NetVocab(CFG)
    a = parse_net_action("connect h0 h2 443")
    d1 = build_graph_model(vocab, CFG, seed=7).predict_delta(_state(), a)
    d2 = build_graph_model(vocab, CFG, seed=7).predict_delta(_state(), a)
    assert d1 == d2


def test_gradients_flow_through_full_stack() -> None:
    """One teacher-forced step produces a finite loss and gradients in every component."""
    vocab = NetVocab(CFG)
    model = build_graph_model(vocab, CFG, seed=0)
    net = model.net
    net.train()
    device = net.device

    state = _state()
    action = parse_net_action("svc_up h1 80")
    # target: the encoded delta (edits + eos)
    from verisim.netoracle import ReferenceNetworkOracle  # local import: torch-free oracle

    oracle = ReferenceNetworkOracle()
    target = encode_target(oracle.step(state, action).delta, vocab)

    g = build_graph(state, action, CFG)
    node, gfeat, a_link, a_flow = graphs_to_tensors([g], device)
    cond, _belief_var = net.encode(node, gfeat, a_link, a_flow, sample=True)

    inp = torch.tensor([[vocab.gen, *target[:-1]]], dtype=torch.long, device=device)
    labels = torch.tensor([target], dtype=torch.long, device=device)
    logits = net.decode_logits(cond, inp)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    )
    assert torch.isfinite(loss)
    loss.backward()
    # gradients reached the encoder (node_in), the RSSM (to_mu), and the decoder head.
    assert net.node_in.weight.grad is not None and net.node_in.weight.grad.abs().sum() > 0
    assert net.to_mu.weight.grad is not None
    assert net.head.weight.grad is not None and net.head.weight.grad.abs().sum() > 0
