# Model card — verisim-cue scorecard — verisim-cue@0.1.0+53b53fefba0d267e

`score_model(model)` runs any host world-model through the ordered computer-use task suite and
returns a **scorecard**: per task, the model's catch rate, the exact faithful ceiling (1.000 by the
ground-truth-labels contract), the gap, and whether the **oracle was load-bearing** for the model on
that task (gap > 0.05). This is the property no oracle-free computer-use bench can
report — it measures not just whether a model succeeds but whether faithfulness was load-bearing for
its success.

## Scorecard schema (per task)
| order | task | keyed dimension | scored |
|---|---|---|---|
| 0 | process-control | procs | catch rate ∈ [0,1], load-bearing if gap > 0.05 |
| 1 | fd-control | fds | catch rate ∈ [0,1], load-bearing if gap > 0.05 |
| 2 | file-integrity | fs | catch rate ∈ [0,1], load-bearing if gap > 0.05 |
| 3 | content-value | fs | catch rate ∈ [0,1], load-bearing if gap > 0.05 |

## Reference scorecards (the scale-law rungs)

Each rung's trained `M_θ` scored through the suite (catch rate per task; `*` = the oracle was load-bearing for that model on that task).

| model (rung) | per-task catch rate | # load-bearing |
|---|---|---|
| `xs` | process 0.84* · fd 0.75* · file 0.38* · content 0.19* | 4 |
| `s` | process 1.00 · fd 0.78* · file 0.12* · content 0.06* | 3 |
| `m` | process 0.97 · fd 0.88* · file 0.41* · content 0.09* | 3 |
| `l` | process 0.97 · fd 0.84* · file 0.44* · content 0.16* | 3 |

## Intended use
Scoring host (shell/file/process) world-models for computer-use faithfulness, and locating *which*
dynamics a given model still needs the oracle for. **Not** for offensive automation (SPEC.md §13).

## Caveats
The faithful ceiling is exact (deterministic oracle); catch rates and load-bearing flags are
comparable only within a fixed manifest hash and at a stated scale. Computer use here is
shell/file/process, not GUI — the oracle-grounded slice.
