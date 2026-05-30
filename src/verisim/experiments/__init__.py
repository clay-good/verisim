"""Experiment entry points (config-driven). M5 ships a baseline sweep; M6 the E1
headline `H_ε(ρ)` curve; M7 the E2 policy and E3 operator comparisons (SPEC-2 §9).
Each experiment is run via its module (`verisim.experiments.e1` / `.e2` / `.e3`)."""

from __future__ import annotations

from .baselines import run_baseline_sweep, sweep_model

__all__ = ["run_baseline_sweep", "sweep_model"]
