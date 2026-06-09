# Model card — reference proposers — verisim-bench@0.1.0+8147113f0407082f

The fidelity ladder the benchmark must stably order (H65). The committed entries are controlled CPU
stand-ins (a per-step-accuracy `α` drafter); the trained flat-transformer and GNN+RSSM arms are
deferred (GPU, `skipif`-guarded, never scored without a checkpoint — the LP7 rule).

| proposer | tier | fidelity α | status |
|---|---|---|---|
| null | floor | 0.00 | shipped (CPU stand-in) |
| learned-lo | learned-lo | 0.65 | shipped (CPU stand-in) |
| learned-mid | learned-mid | 0.80 | shipped (CPU stand-in) |
| learned-hi | learned-hi | 0.92 | shipped (CPU stand-in) |
| oracle-ceiling | ceiling | 1.00 | shipped (CPU stand-in) |

**Intended use:** discriminative-validity and rank-stability measurement of the benchmark.
**Caveat:** the stand-ins isolate the benchmark's *discriminative* behavior (does it order fidelity
stably?); absolute `H_ε` magnitudes for the real arms require the trained checkpoints.
