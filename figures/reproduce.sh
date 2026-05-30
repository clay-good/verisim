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

echo "== done: figures/{e1_curve,e2_policies,e3_operators}.{png,csv} =="
