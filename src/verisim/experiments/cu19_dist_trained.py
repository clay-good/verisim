"""CU19 / H112 -- the trained distributed arm: does a real learned M_θ track the medium?

Trains (once, then freezes) a flat distributed ``M_θ`` on the *same* workload distribution CU18
evaluates on (:func:`~verisim.acd.dist_targeting.make_dist_workload`, train seeds disjoint from the
eval battery), then runs the CU19 belief-rollout closed loop
(:func:`~verisim.acd.closed_loop_dist.run_cu19`): the staleness-drift probe (the CU8 analogue) + the
four-schedule targeting comparison (the CU5-net analogue), on the real model.

Torch is imported lazily inside the train/load functions (LP7: the torch-free core lives in
``acd/closed_loop_dist.py``; CI never trains). The checkpoint is frozen under
``runs/flagship/dist-l`` and reused across runs -- do not retrain unnecessarily.

    python -m verisim.experiments.cu19_dist_trained --out runs/flagship/dist-l
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.acd.closed_loop_dist import CU19Result, cu19_verdict, run_cu19, write_csv
from verisim.acd.dist_targeting import CU18Config, make_dist_workload
from verisim.distoracle.reference import ReferenceDistOracle


@dataclass(frozen=True)
class CU19TrainConfig:
    """The frozen dist flagship: arch + the make_dist_workload-matched training distribution."""

    name: str = "flagship-dist-l"
    train_seeds: tuple[int, ...] = tuple(range(200, 224))  # disjoint from the eval battery (9000+)
    train_steps_per_traj: int = 40
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    train_iters: int = 700
    lr: float = 3e-3
    model_seed: int = 0
    max_int: int = 256

    @property
    def eval(self) -> CU18Config:
        return CU18Config()  # objects=("cfg","a","b"), horizon 48, 200 deployments


def _build_examples(config: CU19TrainConfig, vocab: Any, oracle: ReferenceDistOracle) -> list[Any]:
    from verisim.distmodel.tokenizer import encode_prompt, encode_target

    dist = config.eval.dist
    sensitive = frozenset(config.eval.sensitive_keys)
    examples: list[Any] = []
    for seed in config.train_seeds:
        start, actions = make_dist_workload(seed, config.train_steps_per_traj, dist, sensitive)
        state = start
        for action in actions:
            result = oracle.step(state, action)
            examples.append(
                (encode_prompt(state, action, vocab), encode_target(result.delta, vocab))
            )
            state = result.state
    return examples


def train_dist_model(config: CU19TrainConfig) -> Any:
    """Train the flat distributed M_θ on the CU18 workload distribution; return the world model."""
    import torch

    from verisim.distmodel import DistVocab, NeuralDistWorldModel, build_dist_dataset  # noqa: F401
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_supervised

    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)  # process-reproducibility (the checkpoint gate depends on it)

    dist = config.eval.dist
    oracle = ReferenceDistOracle(dist)
    vocab = DistVocab(dist, max_int=config.max_int)
    examples = _build_examples(config, vocab, oracle)
    block_size = max(len(p) + len(t) for p, t in examples) + 8  # tight (no 512 floor -> cheaper)
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab), block_size=block_size,
            n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
        )
    )
    train_supervised(
        model, examples, vocab.pad, steps=config.train_iters, lr=config.lr, seed=config.model_seed
    )
    model.eval()
    return NeuralDistWorldModel(model, vocab)


def _manifest(config: CU19TrainConfig, block_size: int) -> dict[str, Any]:
    return {
        "name": config.name,
        "arch": {
            "vocab_size": None,  # filled at save time
            "block_size": block_size,
            "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
        },
        "training": {
            "train_seeds": list(config.train_seeds),
            "train_steps_per_traj": config.train_steps_per_traj,
            "train_iters": config.train_iters, "lr": config.lr,
            "model_seed": config.model_seed, "max_int": config.max_int,
        },
    }


def save_checkpoint(world_model: Any, config: CU19TrainConfig, directory: str | Path) -> Path:
    import torch

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(world_model.model.state_dict(), out / "model.pt")
    manifest = _manifest(config, world_model.model.config.block_size)
    manifest["arch"]["vocab_size"] = world_model.model.config.vocab_size
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def load_dist_model(directory: str | Path, *, max_int: int = 256) -> Any:
    import torch

    from verisim.distmodel import DistVocab, NeuralDistWorldModel
    from verisim.model.transformer import GPT, GPTConfig

    out = Path(directory)
    arch = json.loads((out / "manifest.json").read_text())["arch"]
    model = GPT(
        GPTConfig(
            vocab_size=arch["vocab_size"], block_size=arch["block_size"],
            n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"],
        )
    )
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    model.eval()
    vocab = DistVocab(CU18Config().dist, max_int=max_int)
    return NeuralDistWorldModel(model, vocab)


def load_or_train(directory: str | Path, config: CU19TrainConfig | None = None) -> Any:
    config = config or CU19TrainConfig()
    if (Path(directory) / "model.pt").exists():
        return load_dist_model(directory, max_int=config.max_int)
    world_model = train_dist_model(config)
    save_checkpoint(world_model, config, directory)
    return world_model


def run(
    directory: str = "runs/flagship/dist-l", csv: str = "runs/cu19_dist_trained.csv"
) -> CU19Result:
    model = load_or_train(directory)
    result = run_cu19(model)
    write_csv(result, csv)
    return result


def _print_verdict(result: CU19Result) -> None:
    v = cu19_verdict(result)
    print(f"\nCU19 -- the trained distributed arm ({result.n_episodes} deployments, "
          f"horizon {result.horizon})")
    print(f"  drift: staleness recall {v['staleness_recall']:.3f}  "
          f"omissions {v['omissions']} vs hallucinations {v['hallucinations']}  "
          f"(omission-biased: {v['drift_is_omission_biased']})")
    print(f"  free breach        {v['free_breach_rate']:.3f}")
    print(f"  model self-target  {v['model_breach_rate']:.3f}  ({v['model_calls']:.2f} calls)  "
          f"fails={v['model_self_targeting_fails']}")
    transfers = not v["write_target_does_not_transfer"]
    print(f"  write_target       {v['write_target_breach_rate']:.3f}  "
          f"({v['write_target_calls']:.2f} calls)  transfers={transfers}")
    print(f"  medium             {v['medium_breach_rate']:.3f} rand / "
          f"{v['medium_adversarial_breach']:.3f} adv  ({v['medium_calls']:.2f} calls)")
    print(f"  full oracle        {v['full_oracle_calls']:.2f} calls   "
          f"medium saving {v['medium_call_saving']:.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="runs/flagship/dist-l")
    parser.add_argument("--csv", type=str, default="runs/cu19_dist_trained.csv")
    args = parser.parse_args()
    result = run(args.out, args.csv)
    _print_verdict(result)


if __name__ == "__main__":
    main()
