# Task card — load-bearing verdicts — verisim-cue@0.1.0+53b53fefba0d267e

What distinguishes `verisim-cue` from every other computer-use benchmark: each task carries a
**faithfulness-load-bearing verdict** — does a faithful predictor (oracle rollout) beat a free one
(`M_θ` rollout) by more than the threshold (0.05)? If yes, the oracle-in-the-loop is
load-bearing for control on that task; if no, the model already gets that dimension right. Swept
across the capacity ladder, the boundary between the two *is* the SPEC-21 load-bearing frontier.

| order | task | keyed dimension | structure↔content | load-bearing verdict |
|---|---|---|---|---|
| 0 | process-control | procs | structure | +0.03 → **not load-bearing** |
| 1 | fd-control | fds | near-structure | +0.16 → **load-bearing** |
| 2 | file-integrity | fs | content | +0.56 → **load-bearing** |
| 3 | content-value | fs | content | +0.84 → **load-bearing** |

Verdicts measured at scale ≈ 110592 params (the top CPU rung).

**Reading:** the gap should rise with the task order (structure→content) and the load-bearing tasks
should be the deeper-content ones — the structure tasks the model already learns faithfully. The
SPEC-21 scale law measures how this frontier *moves* as capacity grows (it recedes structural-first,
leaving an irreducible content residue, H87/H88).
