"""Autoresearch-style outer search loop (SPEC-2 §17.5 automation).

A deterministic *keep-if-better* ratchet over the config space, gated on an
**oracle-grounded** score rather than a statistical proxy. Inspired by Karpathy's
``autoresearch`` (edit code -> fixed-budget train -> score one number -> keep if it
improves, else roll back), but with verisim's twist: the "did we improve?" gate is
the deterministic oracle (reality), so the comparison is ground truth, not held-out
loss. This automates the §17.5 open problem -- co-tuning model/training/difficulty
until the clean (ρ=0) floor lifts -- as a reproducible ratchet instead of manual
tuning.

The loop lives in :mod:`verisim.auto.search` and is run via
``python -m verisim.auto.search --config configs/auto.json``.
"""

from __future__ import annotations
