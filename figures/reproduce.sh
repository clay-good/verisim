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

echo "== EN9: oracle hard-negative contrastive (SPEC-8 OG4, H25/H5; writes CSV+PNG directly) =="
python -m verisim.experiments.en9 --out figures/en9_contrastive.csv

echo "== EN8 scale-up: collapse/residual gaps vs world size, bootstrap CIs (SPEC-8 OG6, SPEC-9) =="
python -m verisim.experiments.en8_scale --world-sizes 5 10 15 --seeds 0 1 2 3 --out figures/en8_scale.csv

echo "== EN9 scale-up: interventional lift vs world size, bootstrap CIs (SPEC-8 OG6, SPEC-9) =="
python -m verisim.experiments.en9_scale --world-sizes 5 10 15 --seeds 0 1 2 3 --out figures/en9_scale.csv

echo "== EN8 capacity frontier: H24/S3 residual gap vs d_model x observed-fraction (SPEC-9 S3) =="
python -m verisim.experiments.en8_capacity --world-size 40 --d-models 16 32 64 \
    --observed-fractions 0.25 0.5 0.75 --seeds 0 1 2 3 --out figures/en8_capacity.csv

echo "== EN9 S2-recovery: does scaling k_negatives restore the H5 lift? (SPEC-9 S2; slower) =="
python -m verisim.experiments.en9_negatives --world-size 100 --d-model 128 \
    --k-negatives 8 16 32 --seeds 0 1 2 --out figures/en9_negatives.csv

echo "== EN7: model-invariance of H_eps(rho) across proposers (SPEC-5 H22; writes CSV+PNG) =="
python -m verisim.experiments.en7 --out figures/en7_invariance.csv

echo "== EN5: online self-healing (TTT) vs frozen on H_eps(rho) (SPEC-5 H7; writes CSV+PNG) =="
python -m verisim.experiments.en5 --out figures/en5_selfheal.csv

echo "== EN6: counterfactual grounding for change-safety, 3-arm volume control (SPEC-5 H5; CSV+PNG) =="
python -m verisim.experiments.en6 --out figures/en6_counterfactual.csv

echo "== LS3 hero: H23 collapse gap at the local-max world N=300 (SPEC-9 LS3; collapse-only; slow) =="
python -m verisim.experiments.en8_scale --collapse-only --world-sizes 300 --d-models 128 \
    --seeds 0 1 2 --out figures/en8_ls3_hero.csv

echo "== EN10: two-oracle grounding — control-plane vs data-plane (SPEC-5 H12; writes CSV+PNG) =="
python -m verisim.experiments.en10 --out figures/en10_two_oracle.csv

echo "== EH1: composed-host H_eps(rho) curve + composition law H13 (SPEC-6 HC6; the prime directive) =="
python -m verisim.experiments.eh1 --config configs/eh1.json \
    --out runs/eh1/host_records.jsonl --comp-out runs/eh1/composition.json
python figures/plot_eh1.py --records runs/eh1/host_records.jsonl \
    --composition runs/eh1/composition.json \
    --out figures/eh1_curve.png --csv figures/eh1_curve.csv \
    --comp-out figures/eh1_composition.png --comp-csv figures/eh1_composition.csv

echo "== EH3: host correction-operator comparison at equal budget (SPEC-6 HC7, §8.3) =="
python -m verisim.experiments.eh3 --config configs/eh3.json --out runs/eh3/records.jsonl
python figures/plot_comparison.py --records runs/eh3/records.jsonl --key operator \
    --out figures/eh3_operators.png --csv figures/eh3_operators.csv

# The larger world x model SCALING SURFACE (SPEC-9 LS2) and HERO instance (LS3) are opt-in (slower):
#   python -m verisim.experiments.en8_scale --world-sizes 25 50 100 200 --d-models 64 128 --seeds 0 1 2 --out figures/en8_surface.csv
#   python -m verisim.experiments.en9_scale --world-sizes 25 50 100 200 --d-models 64 128 --seeds 0 1 2 --out figures/en9_surface.csv

echo "== done: figures/{e1_curve,e2_policies,e3_operators,calibration,e4_ablation,objective,representation,auto_search,en1_curve,en2_policies,en3_operators,en4_graph_vs_flat,en8_grounding,en9_contrastive,en8_scale,en9_scale,en8_capacity,en9_negatives,en7_invariance,en5_selfheal,en6_counterfactual,en8_ls3_hero,en10_two_oracle,eh1_curve,eh1_composition,eh3_operators}.{png,csv} =="
