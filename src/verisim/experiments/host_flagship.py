"""HFL0 -- train and freeze the HOST flagship M_θ (SPEC-20 §7 host fork; the boundary's other side).

The SPEC-20 network thread established, across six formulations, that oracle-grounded
faithfulness is *not* load-bearing for control in the network world -- a flat model learns the
control-relevant dynamics faithfully (`host_down` 0% drift) and the network horizon is
long (`H_free ≈ 18`). The boundary that finding drew: faithfulness-for-control should appear only
where the model is *bad* at the control-relevant dynamics. SPEC-10 HS2 already showed the **host**
world is exactly that harder regime -- the same capacity ladder there tops out at `H_free ≈ 5` (3-5×
lower than the network), because the composed process/fd/fs/exit dynamics are richer per step.

HFL0 is the host analogue of FL0: it trains a single flat host `M_θ` at the HS2 frontier scale and
**freezes** it, so the host usefulness experiments (process-containment) run against
one real, materially-less-faithful model. The regression gate is FL0's: a reload must reproduce the
manifest `H_free` bit-for-bit, and that `H_free` must land in the SPEC-10 HS2 host plausibility band
(far lower than the network's). Reuses the HS2 harness verbatim
([`horizon_host_scaling`](./horizon_host_scaling.py)); HFL0 adds only the
single-model freeze/reload/gate lifecycle. CPU-local; CI runs the smoke instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.hostmodel import HostVocab, build_host_dataset
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle

from .horizon_host_scaling import (
    HostHorizonScalingConfig,
    _evaluate,
    _train_scaled,
)
from .horizon_scaling import ModelScale

if TYPE_CHECKING:
    from verisim.hostmodel import NeuralHostWorldModel

# The SPEC-10 HS2 host frontier capacity (the `l` cell). The host `H_free` is far lower than the
# network's (~5 vs ~18) -- the harder world is the point (this is where faithfulness may bite).
HOST_FRONTIER_SCALE = ModelScale(label="l", n_embd=192, n_layer=3, n_head=4, train_steps=4000)
HOST_FRONTIER_DATA_SEEDS = 48
# A deliberately wide band: a corrupt/under-trained checkpoint reads ~0; the real host `l` a few
# steps. We only need to reject impostors, not pin the exact (seed-varying) value.
HOST_FRONTIER_H_FREE_ID_BAND = (1.5, 20.0)


@dataclass(frozen=True)
class HostFlagshipConfig:
    """One trained host model -- a single (capacity, data, seed) point, not a sweep."""

    name: str = "flagship-host-l"
    scale: ModelScale = HOST_FRONTIER_SCALE
    data_seeds: int = HOST_FRONTIER_DATA_SEEDS
    train_seed: int = 0
    train_steps_per_traj: int = 60
    block_size: int = 256
    batch_size: int = 64
    lr: float = 3e-3
    num_threads: int = 1
    train_driver: str = "forky"
    eval_driver: str = "forky"
    eval_driver_hard: str = "adversarial"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    eval_steps: int = 48
    one_step_seeds: tuple[int, ...] = (200, 201, 202)
    one_step_steps: int = 40
    epsilon: float = 0.0

    def _hs_config(self) -> HostHorizonScalingConfig:
        return HostHorizonScalingConfig(
            name=self.name, scales=(self.scale,), seeds=(self.train_seed,),
            train_driver=self.train_driver, train_seeds=tuple(range(self.data_seeds)),
            train_steps_per_traj=self.train_steps_per_traj, block_size=self.block_size,
            batch_size=self.batch_size, lr=self.lr, num_threads=self.num_threads,
            eval_driver=self.eval_driver, eval_driver_hard=self.eval_driver_hard,
            eval_seeds=self.eval_seeds, eval_steps=self.eval_steps,
            one_step_seeds=self.one_step_seeds, one_step_steps=self.one_step_steps,
            epsilon=self.epsilon,
        )

    @staticmethod
    def smoke() -> HostFlagshipConfig:
        return HostFlagshipConfig(
            name="flagship-host-smoke",
            scale=ModelScale(label="xs", n_embd=32, n_layer=1, train_steps=300),
            data_seeds=4, train_steps_per_traj=20, eval_seeds=(100, 101), eval_steps=16,
            one_step_seeds=(200,), one_step_steps=10,
        )

    @staticmethod
    def from_json_file(path: str | Path) -> HostFlagshipConfig:
        d = json.loads(Path(path).read_text())
        b = HostFlagshipConfig()
        s = d.get("scale", {})
        scale = (
            ModelScale(
                label=s["label"], n_embd=s["n_embd"], n_layer=s["n_layer"],
                n_head=s.get("n_head", 4), train_steps=s.get("train_steps", 4000),
            )
            if s
            else b.scale
        )
        return HostFlagshipConfig(
            name=d.get("name", b.name), scale=scale, data_seeds=d.get("data_seeds", b.data_seeds),
            train_seed=d.get("train_seed", b.train_seed),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            block_size=d.get("block_size", b.block_size),
            batch_size=d.get("batch_size", b.batch_size),
            lr=d.get("lr", b.lr), num_threads=d.get("num_threads", b.num_threads),
            eval_steps=d.get("eval_steps", b.eval_steps), epsilon=d.get("epsilon", b.epsilon),
        )


@dataclass(frozen=True)
class HostFlagshipCheckpoint:
    world_model: NeuralHostWorldModel
    manifest: dict[str, Any]

    @property
    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = self.manifest["metrics"]
        return metrics


def train_host_flagship(
    config: HostFlagshipConfig | None = None, *, oracle: HostOracle | None = None
) -> tuple[NeuralHostWorldModel, dict[str, float]]:
    """Train the single host flagship model; measure its frozen-time `p` / `H_free` (id+ood)."""
    config = config or HostFlagshipConfig()
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host)
    hs = config._hs_config()
    examples = build_host_dataset(
        oracle, vocab, host, driver=config.train_driver, seeds=tuple(range(config.data_seeds)),
        n_steps=config.train_steps_per_traj,
    )
    world_model = _train_scaled(hs, config.scale, config.train_seed, vocab, examples)
    metrics: dict[str, float] = {"n_train": float(len(examples))}
    for regime, driver in (("id", config.eval_driver), ("ood", config.eval_driver_hard)):
        for base_metric, value in _evaluate(world_model, hs, oracle, host, driver).items():
            metrics[f"{base_metric}_{regime}"] = value
    return world_model, metrics


def _manifest(config: HostFlagshipConfig, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "spec": "SPEC-20 HFL0 (host flagship)",
        "name": config.name,
        "frontier_reference": "SPEC-10 HS2 host l (the harder world; H_free ~5 vs network ~18)",
        "arch": {
            "kind": "flat-transformer",
            "vocab_size": len(HostVocab(DEFAULT_HOST_CONFIG)),
            "block_size": config.block_size,
            "n_layer": config.scale.n_layer,
            "n_head": config.scale.n_head,
            "n_embd": config.scale.n_embd,
            "params": config.scale.params,
        },
        "training": {
            "scale_label": config.scale.label, "data_seeds": config.data_seeds,
            "train_steps": config.scale.train_steps, "train_seed": config.train_seed,
            "driver": config.train_driver,
        },
        "metrics": metrics,
    }


def save_checkpoint(
    world_model: NeuralHostWorldModel, config: HostFlagshipConfig, metrics: dict[str, float],
    directory: str | Path,
) -> Path:
    """Freeze the host flagship: ``model.pt`` (state_dict) + ``manifest.json``."""
    import torch

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(world_model.model.state_dict(), out / "model.pt")
    (out / "manifest.json").write_text(json.dumps(_manifest(config, metrics), indent=2) + "\n")
    return out


def load_checkpoint(directory: str | Path) -> HostFlagshipCheckpoint:
    """Rebuild the host flagship GPT from ``manifest.json`` + ``model.pt`` as a host model."""
    import torch

    from verisim.hostmodel import NeuralHostWorldModel
    from verisim.model.transformer import GPT, GPTConfig

    out = Path(directory)
    manifest = json.loads((out / "manifest.json").read_text())
    arch = manifest["arch"]
    model = GPT(
        GPTConfig(
            vocab_size=arch["vocab_size"], block_size=arch["block_size"],
            n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"],
        )
    )
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    model.eval()
    vocab = HostVocab(DEFAULT_HOST_CONFIG)
    return HostFlagshipCheckpoint(NeuralHostWorldModel(model, vocab), manifest)


def verify_checkpoint(
    directory: str | Path, *, config: HostFlagshipConfig | None = None,
    oracle: HostOracle | None = None, tol: float = 1e-6,
) -> dict[str, Any]:
    """Regression gate: reload reproduces the manifest `H_free` and lands in the HS2 host band."""
    config = config or HostFlagshipConfig()
    oracle = oracle or ReferenceHostOracle()
    ckpt = load_checkpoint(directory)
    host = DEFAULT_HOST_CONFIG
    hs = config._hs_config()
    reeval: dict[str, float] = {}
    for regime, driver in (("id", config.eval_driver), ("ood", config.eval_driver_hard)):
        for base_metric, value in _evaluate(ckpt.world_model, hs, oracle, host, driver).items():
            reeval[f"{base_metric}_{regime}"] = value
    frozen = ckpt.metrics
    reload_ok = all(abs(reeval[k] - frozen[k]) <= tol for k in ("h_free_id", "h_free_ood"))
    lo, hi = HOST_FRONTIER_H_FREE_ID_BAND
    band_ok = lo <= frozen["h_free_id"] <= hi
    return {
        "ok": reload_ok and band_ok,
        "reload_deterministic": reload_ok,
        "in_frontier_band": band_ok,
        "frozen_h_free_id": frozen["h_free_id"],
        "reeval_h_free_id": reeval["h_free_id"],
        "band": list(HOST_FRONTIER_H_FREE_ID_BAND),
    }


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="HFL0 -- freeze the HOST flagship M_θ (SPEC-20).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="runs/flagship/host-l")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = (
        HostFlagshipConfig.smoke() if args.smoke
        else HostFlagshipConfig.from_json_file(args.config) if args.config
        else HostFlagshipConfig()
    )
    print(
        f"training host flagship '{config.name}' "
        f"({config.scale.label}, {config.scale.params:,} params)..."
    )
    world_model, metrics = train_host_flagship(config)
    path = save_checkpoint(world_model, config, metrics, args.out)
    print(
        f"froze host checkpoint -> {path}\n"
        f"  [id]  p={metrics['one_step_acc_id']:.3f} H_free={metrics['h_free_id']:.2f}\n"
        f"  [ood] p={metrics['one_step_acc_ood']:.3f} H_free={metrics['h_free_ood']:.2f}"
    )
    verdict = verify_checkpoint(args.out, config=config)
    print(f"regression gate: {'PASS' if verdict['ok'] else 'FAIL'}  {verdict}")


if __name__ == "__main__":  # pragma: no cover
    main()
