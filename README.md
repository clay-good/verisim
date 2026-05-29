# Verisim

**Oracle-grounded, neuro-symbolic world models of computer environments.**

Generative world models (Genie 3, V-JEPA 2, Cosmos) all hit the same wall:
long-horizon error accumulation and faithfulness, with no cheap way to detect or
correct drift, because physical and visual worlds have no ground-truth oracle.
Verisim's claim is that **computer environments are the exception** — filesystems,
processes, networks, and APIs are digital, deterministic, and fully checkable, so
a deterministic oracle can be placed in the loop to bound a neural world model's
drift. Verisim builds that loop and measures the central tradeoff nobody else can
measure: **how much oracle consultation buys how much faithful horizon.**

- **The science:** [SPEC.md](./SPEC.md) — why the project exists, what it claims,
  and how we would know if we were wrong.
- **The build:** [SPEC-2.md](./SPEC-2.md) — the concrete v0 environment, oracle,
  model, metrics, baselines, repo layout, and milestones.
- **Semantics:** [docs/semantics.md](./docs/semantics.md) — the normative
  description of the v0 shell/filesystem command semantics (paired with the
  reference oracle, which is the executable truth).

## Status

Pre-experiment (v0). The deterministic foundation — milestones **M0–M3** of
SPEC-2 §13 — is implemented and tested:

| Milestone | What | Status |
|-----------|------|--------|
| **M0** | Env (state, command grammar, canonical serialization) + `ReferenceOracle` | ✅ |
| **M1** | `Delta` types, `apply(state, delta)`, delta↔serialization | ✅ |
| **M2** | Drivers, trajectory JSONL, versioned manifests/splits | ✅ |
| **M3** | Divergence `d(s,ŝ)`, faithful horizon `H_ε`, run-record schema | ✅ |
| M4 | Neural world model + supervised training (needs PyTorch/GPU) | ⬜ |
| M5 | Propose–verify–correct loop (policies + operators) | ⬜ |
| M6 | E1 — the `H_ε(ρ)` curve (the v0 result) | ⬜ |

M0–M3 is the deterministic core; it has **no runtime dependencies** and needs no
GPU. The neural model (M4+) adds PyTorch and is intentionally kept out of the base
install.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # property tests, semantics goldens, metric tests
ruff check .      # lint
mypy              # strict type-check
```

```python
from verisim.env import State, parse_action
from verisim.oracle import ReferenceOracle
from verisim.delta import apply

oracle = ReferenceOracle()
state = State.empty()
for cmd in ["mkdir /a", "write /a/f alpha", "mv /a /b", "cat /b/f"]:
    result = oracle.step(state, parse_action(cmd))
    # apply(state, result.delta) == result.state, by construction (the M1 invariant)
    assert apply(state, result.delta).fs == result.state.fs
    state = result.state
```

## Layout

See [SPEC-2.md §10](./SPEC-2.md). The implemented packages live under
[src/verisim/](src/verisim/): `env/`, `oracle/`, `delta/`, `data/`, `metrics/`.
`model/`, `train/`, `loop/`, and `experiments/` are placeholders for M4+.

## License & posture

MIT (see [LICENSE](./LICENSE)). This is a research repo: **no telemetry, no
network calls at runtime, no commercial path.** The framing and downstream agents
are defensive; see [SPEC.md §13](./SPEC.md) for the ethics and dual-use posture.

Author: Clay Good.
