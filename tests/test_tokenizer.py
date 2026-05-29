"""Tokenizer + grammar tests (SPEC-2 §16): round-trip identity and grammar validity."""

from __future__ import annotations

import random

import pytest

# The vocab/tokenizer/grammar are pure Python, but they live in the `verisim.model`
# package whose __init__ imports the torch-based transformer; skip if torch absent.
pytest.importorskip("torch")

from verisim.data.drivers import DRIVERS, Driver
from verisim.env import DEFAULT_CONFIG, State, parse_action
from verisim.model.grammar import DeltaGrammar
from verisim.model.tokenizer import (
    TokenizeError,
    encode_prompt,
    encode_target,
    greedy_decompose,
    parse_target,
)
from verisim.model.vocab import Vocab
from verisim.oracle import ReferenceOracle

VOCAB = Vocab(DEFAULT_CONFIG)


def test_greedy_decompose_is_unique_over_prefix_free_vocab():
    # "alpha"+"beta" and "omega"+"alpha" decompose back to their tokens.
    assert greedy_decompose("alphabeta", VOCAB) == [
        VOCAB.content_to_id["alpha"],
        VOCAB.content_to_id["beta"],
    ]
    assert greedy_decompose("", VOCAB) == []
    with pytest.raises(TokenizeError):
        greedy_decompose("notavocabword", VOCAB)


@pytest.mark.parametrize("driver_name", DRIVERS)
def test_target_roundtrip_and_grammar_acceptance(driver_name: str):
    """parse_target(encode_target(delta, stdout)) == (delta, stdout), and every
    encoded target is accepted by the grammar automaton."""
    oracle = ReferenceOracle()
    grammar = DeltaGrammar(VOCAB)
    driver = Driver(name=driver_name, config=DEFAULT_CONFIG, rng=random.Random(13))
    state = State.empty()
    for _ in range(120):
        result = oracle.step(state, driver.sample(state))
        target = encode_target(result.delta, result.stdout, VOCAB)

        delta2, stdout2 = parse_target(target, VOCAB)
        assert delta2 == result.delta
        assert stdout2 == result.stdout

        stack = grammar.start()
        for token in target:
            assert token in grammar.allowed(stack)
            stack = grammar.advance(stack, token)
        assert grammar.is_accept(stack)

        state = result.state


def test_prompt_omits_stdout_hash_but_keeps_exit_code():
    # The model input serializes exit_code (a token) but not the stdout hash.
    state = ReferenceOracle().step(State.empty(), parse_action("cat /nope")).state
    prompt = encode_prompt(state, parse_action("ls /"), VOCAB)
    assert VOCAB.exit_to_id[state.last.exit_code] in prompt
    # No token encodes a raw hash; the prompt is entirely within the closed vocab.
    assert all(0 <= t < len(VOCAB) for t in prompt)


def test_grammar_rejects_out_of_place_tokens():
    grammar = DeltaGrammar(VOCAB)
    stack = grammar.start()
    # At the top level, a name token (mid-path) is not allowed.
    name_id = next(iter(VOCAB.name_ids))
    assert name_id not in grammar.allowed(stack)
    assert VOCAB.eos in grammar.allowed(stack)
