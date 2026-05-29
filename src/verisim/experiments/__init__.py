"""Experiment entry points (config-driven). M5 ships a baseline sweep; the E1
headline curve (SPEC-2 §9) lands at M6 with the learned model."""

from __future__ import annotations

from .baselines import run_baseline_sweep, sweep_model

__all__ = ["run_baseline_sweep", "sweep_model"]
