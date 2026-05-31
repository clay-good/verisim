"""Tests for bits-to-correct (SPEC-2.1 §3): zero-iff-equal, monotone, multiset-aware."""

from verisim.delta.edits import Create, Delete, Delta, Modify, SetResult
from verisim.env.state import Dir, File, content_hash
from verisim.metrics.bits import bits_to_correct, correction_symbols, edit_symbols


def test_zero_iff_equal():
    delta: Delta = [
        Create("/a", Dir()),
        Create("/a/b", File("alpha")),
        SetResult(0, content_hash("")),
    ]
    assert bits_to_correct(delta, list(delta)) == 0.0
    assert correction_symbols(delta, list(delta)) == 0


def test_positive_when_different():
    assert bits_to_correct([Create("/a", Dir())], []) > 0.0


def test_monotone_in_residual():
    true = [Create("/a", Dir()), Create("/a/b", File("alpha"))]
    near = [Create("/a", Dir())]  # one edit missing
    assert bits_to_correct([], true) > bits_to_correct(near, true) > 0.0


def test_invented_edit_counts_as_residual():
    true = [Create("/a", Dir())]
    extra: Delta = [Create("/a", Dir()), Delete("/zzz")]  # a hallucinated extra edit
    assert correction_symbols(extra, true) == edit_symbols(Delete("/zzz"))


def test_multiset_multiplicity():
    edit = Modify("/a", "alpha")
    # one extra copy of the same edit is one residual edit's worth of correction.
    assert correction_symbols([edit, edit], [edit]) == edit_symbols(edit)


def test_longer_content_costs_more():
    short = Modify("/a", "alpha")
    long = Modify("/a", "alphabeta")
    assert edit_symbols(long) > edit_symbols(short)
