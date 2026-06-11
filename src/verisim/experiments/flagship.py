"""FL0 -- train and freeze the flagship network world model (SPEC-19 §5, milestone FL0).

Every method spec to date proved its mechanism on a *controlled stand-in* and deferred the
trained-``M_θ`` arm (SPEC-19 §1). SPEC-10 then scaled the *floor* (``H_free`` at ``ρ=0``) on a real
trained model but never carried the consultation curve or composed the methods. FL0 is the first
brick of the fix: it un-defers the trained model *once* -- trains a single flat ``M_θ`` at the
SPEC-10 HS1.3 **compute-optimal frontier** (the ``l@9.6k`` cell: ``n_embd=192``, ``n_layer=3``,
~9,600 transitions, the program-best ~19 id / ~29 ood free-running steps,
[`horizon_joint_scaling`](./horizon_joint_scaling.py)) -- and **freezes it** as a versioned,
reloadable checkpoint with a manifest (the model-card seed SPEC-18 PB-pack extends). Everything
downstream of SPEC-19 (FL1's headline curve, FL2's composition ablation, FL3's goal-horizon) runs
against *this one frozen checkpoint*, so the flagship figure is a property of a single real model,
not a fresh re-train per experiment.

The regression gate (``verify_checkpoint``) is what makes "frozen" mean something: a reloaded
checkpoint must reproduce its manifest ``H_free`` **bit-for-bit** (the loop is deterministic at
``num_threads=1``), and that ``H_free`` must fall inside the SPEC-10 HS1.3 ``l``-cell plausibility
band -- so a silently-corrupted or mis-trained checkpoint cannot pass as the flagship.

Reuses the HS1 harness verbatim (``ModelScale``, ``HorizonScalingConfig``, ``_train_scaled``,
``_evaluate``): FL0 invents no training loop and no metric (SPEC-19 §2 -- no new world, oracle, or
method), only the *single-model* lifecycle (train one, freeze it, reload it, gate it). CPU-local; CI
runs only the smoke instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netmodel import NetVocab, build_net_dataset
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .en1 import EN1Config
from .horizon_scaling import HorizonScalingConfig, ModelScale, _evaluate, _train_scaled

if TYPE_CHECKING:
    from verisim.netmodel import NeuralNetworkWorldModel

# The SPEC-10 HS1.3 compute-optimal frontier cell (configs/horizon_joint_scaling.json, the `l`
# point): the program-best free-running horizon. This is THE flagship capacity/data operating point.
FRONTIER_SCALE = ModelScale(label="l", n_embd=192, n_layer=3, n_head=4, train_steps=5000)
FRONTIER_DATA_SEEDS = 96  # × train_steps_per_traj=100 ≈ 9,600 transitions (the "@9.6k")
# The HS1.3 `l@9.6k` committed result (SPEC-10 §4.4): id 19.2 / ood 28.75, 3-seed mean. A single
# flagship seed varies around this; the gate's plausibility band is deliberately wide (a corrupted
# checkpoint reads ~0, an under-trained one reads ~2 -- both fall far outside, all the gate needs).
FRONTIER_H_FREE_ID_BAND = (8.0, 40.0)


@dataclass(frozen=True)
class FlagshipConfig:
    """One trained model -- a single (capacity, data, seed) point, not a sweep.

    ``train_seed`` picks the one checkpoint we freeze (HS1.3 averaged over seeds 0-2; the flagship
    is one concrete model, ``train_seed=0`` by default). Eval params mirror the HS1.3 config so the
    frozen ``H_free`` is measured on the same regime the frontier number was.
    """

    name: str = "flagship-net-l"
    scale: ModelScale = FRONTIER_SCALE
    data_seeds: int = FRONTIER_DATA_SEEDS
    train_seed: int = 0
    train_steps_per_traj: int = 100
    block_size: int = 128
    batch_size: int = 64
    lr: float = 3e-3
    num_threads: int = 1  # 1 = bit-deterministic reload (the gate depends on this)
    eval_driver: str = "weighted"
    eval_driver_hard: str = "adversarial"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    eval_steps: int = 96
    one_step_seeds: tuple[int, ...] = (200, 201, 202)
    one_step_steps: int = 40
    epsilon: float = 0.0

    def _hs_config(self) -> HorizonScalingConfig:
        """Adapt to the HS1 harness's config (so ``_train_scaled``/``_evaluate`` run unchanged)."""
        base = EN1Config(
            name=f"{self.name}-data",
            train_driver=self.eval_driver,
            train_seeds=tuple(range(self.data_seeds)),
            train_steps_per_traj=self.train_steps_per_traj,
            block_size=256,
            lr=self.lr,
        )
        return HorizonScalingConfig(
            name=self.name, base=base, scales=(self.scale,), seeds=(self.train_seed,),
            block_size=self.block_size, batch_size=self.batch_size, lr=self.lr,
            num_threads=self.num_threads, eval_driver=self.eval_driver,
            eval_driver_hard=self.eval_driver_hard, eval_seeds=self.eval_seeds,
            eval_steps=self.eval_steps, one_step_seeds=self.one_step_seeds,
            one_step_steps=self.one_step_steps, epsilon=self.epsilon,
        )

    @staticmethod
    def smoke() -> FlagshipConfig:
        """A tiny, seconds-scale instance for CI/tests: same lifecycle, trivial model + data."""
        return FlagshipConfig(
            name="flagship-net-smoke",
            scale=ModelScale(label="xs", n_embd=32, n_layer=1, train_steps=300),
            data_seeds=4, train_steps_per_traj=20, eval_seeds=(100, 101), eval_steps=16,
            one_step_seeds=(200,), one_step_steps=10,
        )

    @staticmethod
    def from_json_file(path: str | Path) -> FlagshipConfig:
        d = json.loads(Path(path).read_text())
        b = FlagshipConfig()
        s = d.get("scale", {})
        scale = (
            ModelScale(
                label=s["label"], n_embd=s["n_embd"], n_layer=s["n_layer"],
                n_head=s.get("n_head", 4), train_steps=s.get("train_steps", 5000),
            )
            if s
            else b.scale
        )
        return FlagshipConfig(
            name=d.get("name", b.name), scale=scale, data_seeds=d.get("data_seeds", b.data_seeds),
            train_seed=d.get("train_seed", b.train_seed),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            block_size=d.get("block_size", b.block_size),
            batch_size=d.get("batch_size", b.batch_size),
            lr=d.get("lr", b.lr), num_threads=d.get("num_threads", b.num_threads),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_driver_hard=d.get("eval_driver_hard", b.eval_driver_hard),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            one_step_seeds=tuple(d.get("one_step_seeds", b.one_step_seeds)),
            one_step_steps=d.get("one_step_steps", b.one_step_steps),
            epsilon=d.get("epsilon", b.epsilon),
        )


@dataclass(frozen=True)
class FlagshipCheckpoint:
    """A reloaded flagship: the world model plus the manifest it was frozen with."""

    world_model: NeuralNetworkWorldModel
    manifest: dict[str, Any]

    @property
    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = self.manifest["metrics"]
        return metrics


def train_flagship(
    config: FlagshipConfig | None = None, *, oracle: NetOracle | None = None
) -> tuple[NeuralNetworkWorldModel, dict[str, float]]:
    """Train the single flagship model and measure its frozen-time ``p`` / ``H_free`` (id + ood)."""
    config = config or FlagshipConfig()
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)
    hs = config._hs_config()
    base = hs.base
    examples = build_net_dataset(
        oracle, vocab, net, driver=base.train_driver, seeds=base.train_seeds,
        n_steps=base.train_steps_per_traj,
    )
    world_model = _train_scaled(hs, config.scale, config.train_seed, vocab, examples)
    metrics: dict[str, float] = {"n_train": float(len(examples))}
    for regime, driver in (("id", config.eval_driver), ("ood", config.eval_driver_hard)):
        for base_metric, value in _evaluate(world_model, hs, oracle, net, driver).items():
            metrics[f"{base_metric}_{regime}"] = value
    return world_model, metrics


def _manifest(config: FlagshipConfig, metrics: dict[str, float]) -> dict[str, Any]:
    """The model-card seed (SPEC-18 PB-pack extends this): everything needed to rebuild + gate."""
    net = DEFAULT_NET_CONFIG
    return {
        "spec": "SPEC-19 FL0",
        "name": config.name,
        "frontier_reference": "SPEC-10 HS1.3 l@9.6k (configs/horizon_joint_scaling.json)",
        "arch": {
            "kind": "flat-transformer",
            "vocab_size": len(NetVocab(net)),
            "block_size": config.block_size,
            "n_layer": config.scale.n_layer,
            "n_head": config.scale.n_head,
            "n_embd": config.scale.n_embd,
            "params": config.scale.params,
        },
        "training": {
            "scale_label": config.scale.label,
            "data_seeds": config.data_seeds,
            "train_steps": config.scale.train_steps,
            "train_steps_per_traj": config.train_steps_per_traj,
            "train_seed": config.train_seed,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "num_threads": config.num_threads,
        },
        "eval": {
            "driver_id": config.eval_driver,
            "driver_ood": config.eval_driver_hard,
            "eval_seeds": list(config.eval_seeds),
            "eval_steps": config.eval_steps,
            "epsilon": config.epsilon,
        },
        "metrics": metrics,
    }


def save_checkpoint(
    world_model: NeuralNetworkWorldModel, config: FlagshipConfig, metrics: dict[str, float],
    directory: str | Path,
) -> Path:
    """Freeze the flagship: ``model.pt`` (state_dict) + ``manifest.json`` (arch, training, metrics).

    The state_dict is sufficient with the manifest's ``arch`` block to rebuild the exact GPT; the
    vocab is reconstructed deterministically from ``DEFAULT_NET_CONFIG`` (no need to serialize it).
    """
    import torch

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(world_model.model.state_dict(), out / "model.pt")
    (out / "manifest.json").write_text(json.dumps(_manifest(config, metrics), indent=2) + "\n")
    return out


def load_checkpoint(directory: str | Path) -> FlagshipCheckpoint:
    """Rebuild the flagship GPT from ``manifest.json`` + ``model.pt``, wrapped as a world model."""
    import torch

    from verisim.model.transformer import GPT, GPTConfig
    from verisim.netmodel import NeuralNetworkWorldModel

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
    vocab = NetVocab(DEFAULT_NET_CONFIG)
    return FlagshipCheckpoint(NeuralNetworkWorldModel(model, vocab), manifest)


def verify_checkpoint(
    directory: str | Path, *, config: FlagshipConfig | None = None, oracle: NetOracle | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Regression gate: a reload must reproduce the manifest ``H_free`` and hit the SPEC-10 band.

    Two checks, both required for ``ok``:
      - **reload determinism** -- re-evaluating the loaded checkpoint reproduces the frozen
        ``h_free_id`` / ``h_free_ood`` within ``tol`` (the loop is deterministic at
        ``num_threads=1``; any drift means a corrupt save/load or a nondeterminism leak);
      - **frontier plausibility** -- the frozen ``h_free_id`` lies in the SPEC-10 HS1.3 ``l``-cell
        band (a corrupted/under-trained checkpoint reads near 0, far outside).
    """
    config = config or FlagshipConfig()
    oracle = oracle or ReferenceNetworkOracle()
    ckpt = load_checkpoint(directory)
    net = DEFAULT_NET_CONFIG
    hs = config._hs_config()

    reeval: dict[str, float] = {}
    for regime, driver in (("id", config.eval_driver), ("ood", config.eval_driver_hard)):
        for base_metric, value in _evaluate(ckpt.world_model, hs, oracle, net, driver).items():
            reeval[f"{base_metric}_{regime}"] = value

    frozen = ckpt.metrics
    reload_ok = all(
        abs(reeval[k] - frozen[k]) <= tol for k in ("h_free_id", "h_free_ood")
    )
    lo, hi = FRONTIER_H_FREE_ID_BAND
    band_ok = lo <= frozen["h_free_id"] <= hi
    return {
        "ok": reload_ok and band_ok,
        "reload_deterministic": reload_ok,
        "in_frontier_band": band_ok,
        "frozen_h_free_id": frozen["h_free_id"],
        "reeval_h_free_id": reeval["h_free_id"],
        "frozen_h_free_ood": frozen["h_free_ood"],
        "reeval_h_free_ood": reeval["h_free_ood"],
        "band": list(FRONTIER_H_FREE_ID_BAND),
    }


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="FL0 -- freeze the flagship M_θ (SPEC-19).")
    parser.add_argument(
        "--config", type=str, default=None, help="FlagshipConfig JSON (default: frontier)"
    )
    parser.add_argument(
        "--out", type=str, default="runs/flagship/net-l", help="checkpoint directory"
    )
    parser.add_argument("--smoke", action="store_true", help="train the tiny CI instance instead")
    args = parser.parse_args()
    config = (
        FlagshipConfig.smoke() if args.smoke
        else FlagshipConfig.from_json_file(args.config) if args.config
        else FlagshipConfig()
    )
    print(
        f"training flagship '{config.name}' "
        f"({config.scale.label}, {config.scale.params:,} params)..."
    )
    world_model, metrics = train_flagship(config)
    path = save_checkpoint(world_model, config, metrics, args.out)
    print(
        f"froze checkpoint -> {path}\n"
        f"  [id]  p={metrics['one_step_acc_id']:.3f} H_free={metrics['h_free_id']:.2f} "
        f"eta={metrics['horizon_efficiency_id']:.2f}\n"
        f"  [ood] p={metrics['one_step_acc_ood']:.3f} H_free={metrics['h_free_ood']:.2f} "
        f"eta={metrics['horizon_efficiency_ood']:.2f}"
    )
    verdict = verify_checkpoint(args.out, config=config)
    print(f"regression gate: {'PASS' if verdict['ok'] else 'FAIL'}  {verdict}")


if __name__ == "__main__":  # pragma: no cover
    main()
