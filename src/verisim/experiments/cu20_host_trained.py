"""CU20 / H113 -- the trained host arm: does a real learned M_θ foresee a corruption?

Loads the frozen host flagship (``runs/flagship/host-l``, SPEC-20 HFL0 -- reused, **no retrain**)
and runs the CU20 teacher-forced closed loop
(:func:`~verisim.acd.closed_loop_host.run_cu20`): the write-drift probe (the CU8 analogue) + the
three-schedule targeting comparison (the CU5-net analogue), on the real model.

Torch is imported lazily inside the load function (LP7: the torch-free core lives in
``acd/closed_loop_host.py``; CI never trains). The checkpoint is frozen and reused -- single-step
``predict_delta`` on horizon-bounded host states is milliseconds, so the run is tractable on CPU
(the old host pathology was the ``imagine`` rollout gate, not single-step prediction).

    python -m verisim.experiments.cu20_host_trained --ckpt runs/flagship/host-l
"""

from __future__ import annotations

import argparse
from typing import Any

from verisim.acd.closed_loop_host import CU20Result, cu20_verdict, run_cu20, write_csv


def load_host_model(directory: str = "runs/flagship/host-l") -> Any:
    """Rebuild the frozen host flagship M_θ as a predict_delta model (torch, gated)."""
    from verisim.experiments.host_flagship import load_checkpoint

    return load_checkpoint(directory).world_model


def run(
    directory: str = "runs/flagship/host-l", csv: str = "runs/cu20_host_trained.csv"
) -> CU20Result:
    model = load_host_model(directory)
    result = run_cu20(model)
    write_csv(result, csv)
    return result


def _print_verdict(result: CU20Result) -> None:
    v = cu20_verdict(result)
    print(f"\nCU20 -- the trained host arm ({result.n_episodes} deployments, "
          f"horizon {result.horizon})")
    print(f"  drift: protected recall {v['protected_recall']:.3f}  "
          f"omissions {v['omissions']} vs hallucinations {v['hallucinations']}  "
          f"(omission-biased: {v['drift_is_omission_biased']})")
    print(f"  free breach        {v['free_breach_rate']:.3f}")
    print(f"  model self-target  {v['model_breach_rate']:.3f}  ({v['model_calls']:.2f} calls)  "
          f"fails={v['model_self_targeting_fails']}")
    print(f"  structure          {v['structure_breach_rate']:.3f} rand / "
          f"{v['structure_adversarial_breach']:.3f} adv  ({v['structure_calls']:.2f} calls)  "
          f"ungameable={v['structure_is_ungameable']}")
    print(f"  full oracle        {v['full_oracle_calls']:.2f} calls   "
          f"structure saving {v['structure_call_saving']:.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default="runs/flagship/host-l")
    parser.add_argument("--csv", type=str, default="runs/cu20_host_trained.csv")
    args = parser.parse_args()
    result = run(args.ckpt, args.csv)
    _print_verdict(result)


if __name__ == "__main__":
    main()
