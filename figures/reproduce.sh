#!/usr/bin/env bash
# Regenerate every v0 figure from its committed config + seeds (SPEC-2 §12, M8).
#
# A stranger with the repo and the [dev,model,viz] extras installed runs this and
# gets figures/e1_curve.*, figures/e2_policies.*, figures/e3_operators.* identical
# to the committed ones (determinism is tested in tests/test_e1.py / test_e2.py /
# test_e3.py). Run-records land under runs/ (git-ignored, regenerable).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== E1: H_eps(rho) curve =="
python -m verisim.experiments.e1 --config configs/e1.json --out runs/e1/records.jsonl
python figures/plot_e1.py --records runs/e1/records.jsonl \
    --out figures/e1_curve.png --csv figures/e1_curve.csv

echo "== E2: consultation-policy comparison =="
python -m verisim.experiments.e2 --config configs/e2.json --out runs/e2/records.jsonl
python figures/plot_comparison.py --records runs/e2/records.jsonl --key policy \
    --out figures/e2_policies.png --csv figures/e2_policies.csv

echo "== E3: correction-operator comparison =="
python -m verisim.experiments.e3 --config configs/e3.json --out runs/e3/records.jsonl
python figures/plot_comparison.py --records runs/e3/records.jsonl --key operator \
    --out figures/e3_operators.png --csv figures/e3_operators.csv

echo "== calibration diagnostic (§7.2) =="
python -m verisim.experiments.calibration --config configs/calibration.json \
    --out runs/calibration/pairs.jsonl
python figures/plot_calibration.py --pairs runs/calibration/pairs.jsonl \
    --out figures/calibration.png --csv figures/calibration.csv

echo "== E4: size/difficulty ablation (trains several models; slower) =="
python -m verisim.experiments.e4 --config configs/e4.json --out runs/e4/records.jsonl
python figures/plot_e4.py --records runs/e4/records.jsonl \
    --out figures/e4_ablation.png --csv figures/e4_ablation.csv

echo "== E4 objective axis: supervised vs. +RLVR (trains + RLVR-tunes; slower) =="
python -m verisim.experiments.objective --config configs/objective.json \
    --out runs/objective/records.jsonl
python figures/plot_objective.py --records runs/objective/records.jsonl \
    --out figures/objective.png --csv figures/objective.csv

echo "== E4 representation axis: delta vs. full-state (trains two models; slower) =="
python -m verisim.experiments.representation --config configs/representation.json \
    --out runs/representation/records.jsonl
python figures/plot_representation.py --records runs/representation/records.jsonl \
    --out figures/representation.png --csv figures/representation.csv

echo "== autoresearch ratchet: oracle-gated keep-if-better config search (§17.5; trains per trial) =="
python -m verisim.auto.search --config configs/auto.json --out runs/auto/log.jsonl
python figures/plot_auto.py --records runs/auto/log.jsonl \
    --out figures/auto_search.png --csv figures/auto_search.csv

echo "== EN1: network H_eps(rho) curve (SPEC-5 NW6; the prime directive) =="
python -m verisim.experiments.en1 --config configs/en1.json --out runs/en1/records.jsonl
python figures/plot_en1.py --records runs/en1/records.jsonl \
    --out figures/en1_curve.png --csv figures/en1_curve.csv

echo "== EN2: network consultation-policy comparison (SPEC-5 NW7, H9) =="
python -m verisim.experiments.en2 --config configs/en2.json --out runs/en2/records.jsonl
python figures/plot_comparison.py --records runs/en2/records.jsonl --key policy \
    --out figures/en2_policies.png --csv figures/en2_policies.csv

echo "== EN3: network correction/belief-operator comparison (SPEC-5 NW7, §8.3) =="
python -m verisim.experiments.en3 --config configs/en3.json --out runs/en3/records.jsonl
python figures/plot_comparison.py --records runs/en3/records.jsonl --key operator \
    --out figures/en3_operators.png --csv figures/en3_operators.csv

echo "== EN4: graph-vs-flat comparison (SPEC-5 NW8, H11; writes CSV+PNG directly) =="
python -m verisim.experiments.en4_graph --graph-iters 1500 --out figures/en4_graph_vs_flat.csv

echo "== EN8: oracle-grounded SSL ablation (SPEC-8 OG3, H23/H24; writes CSV+PNG directly) =="
python -m verisim.experiments.en8 --out figures/en8_grounding.csv

echo "== done: figures/{e1_curve,e2_policies,e3_operators,calibration,e4_ablation,objective,representation,auto_search,en1_curve,en2_policies,en3_operators,en4_graph_vs_flat,en8_grounding}.{png,csv} =="
