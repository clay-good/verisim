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

echo "== EH2: host consultation-policy comparison — flat entropy vs factored belief-variance (HC7, H9) =="
python -m verisim.experiments.eh2 --config configs/eh2.json --out runs/eh2/records.jsonl
python figures/plot_comparison.py --records runs/eh2/records.jsonl --key label \
    --out figures/eh2_policies.png --csv figures/eh2_policies.csv

echo "== EH3: host correction-operator comparison at equal budget (SPEC-6 HC7, §8.3) =="
python -m verisim.experiments.eh3 --config configs/eh3.json --out runs/eh3/records.jsonl
python figures/plot_comparison.py --records runs/eh3/records.jsonl --key operator \
    --out figures/eh3_operators.png --csv figures/eh3_operators.csv

echo "== EH4: factored interaction-graph vs flat M_θ (SPEC-6 HC4 incr-2; the H13 follow-up; CSV+PNG) =="
python -m verisim.experiments.eh4 --config configs/eh4.json --out figures/eh4_factored_vs_flat.csv

echo "== EH4-drift: §6.3 drift-lever ablation on the factored arm — noise / self-forcing (HC7) =="
python -m verisim.experiments.eh4_drift --config configs/eh4_drift.json --out figures/eh4_drift.csv

echo "== EH-H14: the concurrency dial — H_eps vs interleaving entropy (SPEC-6 §3.4, H14; CSV+PNG) =="
python -m verisim.experiments.eh_h14 --config configs/eh_h14.json --out figures/eh_h14_interleaving.csv

echo "== EH-H14-scale: does the concurrency collapse steepen with thread count? (SPEC-6 §3.4; CSV+PNG) =="
python -m verisim.experiments.eh_h14_scale --config configs/eh_h14_scale.json --out figures/eh_h14_scale.csv

echo "== EH7: model-invariance of the composed-host H_eps(rho) across proposers (SPEC-6 H22; CSV+PNG) =="
python -m verisim.experiments.eh7 --config configs/eh7.json --out figures/eh7_invariance.csv

echo "== EH8: privilege-faithfulness — does the model get security-critical denials right? (§9.4) =="
python -m verisim.experiments.eh8_privilege --config configs/eh8_privilege.json --out figures/eh8_privilege.csv

echo "== EH6: two-oracle — the privilege invariant is redundant but cheaper + decision-sufficient (H12) =="
python -m verisim.experiments.eh6_two_oracle --config configs/eh6_two_oracle.json --out figures/eh6_two_oracle.csv

echo "== EH-H13-scale: does the composition coupling deepen with concurrency width? (H13 × H14) =="
python -m verisim.experiments.eh_h13_scale --config configs/eh_h13_scale.json --out figures/eh_h13_scale.csv

echo "== EH9: the denial-weighted objective — does oversampling denials close the EH8 gap? (§9.4) =="
python -m verisim.experiments.eh9_denial_weighted --config configs/eh9_denial_weighted.json \
    --out figures/eh9_denial_weighted.csv

echo "== EH6 (counterfactual): does free oracle counterfactual replay train intervention fidelity? (H16) =="
python -m verisim.experiments.eh6_counterfactual --config configs/eh6_counterfactual.json \
    --out figures/eh6_counterfactual.csv

echo "== EH-stream: does the experience stream beat the batch at equal compute? + plasticity (H15 / HW-4) =="
python -m verisim.experiments.eh_stream --config configs/eh_stream.json --out figures/eh_stream.csv

echo "== EH5: smart which-subsystem policy π_w — uncertainty vs fixed/round-robin (HC7, §8.2) =="
python -m verisim.experiments.eh5 --config configs/eh5.json --out runs/eh5/records.jsonl
python figures/plot_comparison.py --records runs/eh5/records.jsonl --key policy \
    --out figures/eh5_subsystem_policy.png --csv figures/eh5_subsystem_policy.csv

echo "== EH5-heads: trained per-subsystem head vs bucketed-entropy π_w + §9.4 calibration (HC7) =="
python -m verisim.experiments.eh5_heads --config configs/eh5_heads.json --out runs/eh5_heads/records.jsonl
python figures/plot_comparison.py --records runs/eh5_heads/records.jsonl --key policy \
    --out figures/eh5_heads.png --csv figures/eh5_heads.csv

# The larger world x model SCALING SURFACE (SPEC-9 LS2) and HERO instance (LS3) are opt-in (slower):
#   python -m verisim.experiments.en8_scale --world-sizes 25 50 100 200 --d-models 64 128 --seeds 0 1 2 --out figures/en8_surface.csv
#   python -m verisim.experiments.en9_scale --world-sizes 25 50 100 200 --d-models 64 128 --seeds 0 1 2 --out figures/en9_surface.csv

echo "== SYNTHESIS: the floor+cliff H_eps(rho) overlaid across all three worlds (reads the curve CSVs) =="
python -m verisim.experiments.synthesis --out figures/synthesis_floor_cliff.csv

echo "== HS1: the faithful-horizon SCALING LAW — H_free vs capacity vs the independence baseline (SPEC-10, H26) =="
echo "   (local sweep, several minutes on CPU; writes the CSV + the two-panel figure directly)"
python -m verisim.experiments.horizon_scaling --config configs/horizon_scaling.json \
    --out figures/horizon_scaling.csv --plot figures/horizon_scaling.png

echo "== HS1.1: the RESOURCED FRONTIER — non-monotone horizon, xl/xxl, the proxy/truth divergence (SPEC-10, §4.2) =="
echo "   (local sweep, ~2.5 h on CPU; 6 capacities × 3 seeds; writes the CSV + the two-panel figure directly)"
python -m verisim.experiments.horizon_scaling --config configs/horizon_scaling_xl.json \
    --out figures/horizon_scaling_xl.csv --plot figures/horizon_scaling_xl.png

echo "== HS1.2: the DATA CROSS-AXIS at fixed xl — starvation vs capacity wall (SPEC-10, §4.3) =="
echo "   (local sweep, ~3 h on CPU; xl × 4 data budgets × 3 seeds; writes the CSV + the two-panel figure)"
python -m verisim.experiments.horizon_data_scaling --config configs/horizon_data_scaling.json \
    --out figures/horizon_data_scaling.csv --plot figures/horizon_data_scaling.png

echo "== HS1.3: the JOINT capacity×data push — the compute-optimal frontier (SPEC-10, §4.4) =="
echo "   (local sweep, ~3 h on CPU; 4-cell compute-optimal ladder × 3 seeds; writes the CSV + figure)"
python -m verisim.experiments.horizon_joint_scaling --config configs/horizon_joint_scaling.json \
    --out figures/horizon_joint_scaling.csv --plot figures/horizon_joint_scaling.png

echo "== HS2: the SCALING LAW re-run on the HOST world — universality across worlds (SPEC-10, §4.5) =="
echo "   (local sweep, ~1 h on CPU; same capacity axis as HS1, 3 seeds; writes the CSV + the figure)"
python -m verisim.experiments.horizon_host_scaling --config configs/horizon_host_scaling.json \
    --out figures/horizon_host_scaling.csv --plot figures/horizon_host_scaling.png

echo "== HS3: the SCALING LAW with the STRUCTURED (graph) arm — is the lift proposer-dependent? (SPEC-10, §4.6) =="
echo "   (local sweep, ~10 min on CPU; same axis as HS1, GNN+RSSM proposer, 3 seeds; writes CSV + figure)"
python -m verisim.experiments.horizon_graph_scaling --config configs/horizon_graph_scaling.json \
    --out figures/horizon_graph_scaling.csv --plot figures/horizon_graph_scaling.png

echo "== HS3 incr 2: the DATA CROSS-AXIS for the graph arm — is its floor data starvation? (SPEC-10, §4.7) =="
echo "   (local sweep, ~5 min on CPU; fixed graph m capacity × 4 data budgets × 3 seeds; writes CSV + figure)"
python -m verisim.experiments.horizon_graph_data_scaling --config configs/horizon_graph_data_scaling.json \
    --out figures/horizon_graph_data_scaling.csv --plot figures/horizon_graph_data_scaling.png

echo "== HS3 incr 3: the WORLD-SIZE CROSS-AXIS for the graph arm — is the ceiling world-size-invariant? (SPEC-10, §4.8) =="
echo "   (local sweep, ~10 min on CPU; fixed graph m capacity × 4 world sizes (5–40 hosts) × 3 seeds; writes CSV + figure)"
python -m verisim.experiments.horizon_graph_world_scaling --config configs/horizon_graph_world_scaling.json \
    --out figures/horizon_graph_world_scaling.csv --plot figures/horizon_graph_world_scaling.png

echo "== HS3 incr 4: the JOINT capacity×world-size push, structured arm — does scaling both lift it? (SPEC-10, §4.10) =="
echo "   (local sweep, ~10 min on CPU; structured ladder s@5h → xl@40h × 3 seeds; writes CSV + figure)"
python -m verisim.experiments.horizon_graph_joint_scaling --config configs/horizon_graph_joint_scaling.json \
    --out figures/horizon_graph_joint_scaling.csv --plot figures/horizon_graph_joint_scaling.png

echo "== HS3-T: the trainer diagnostic — is the graph p plateau a flat-LR artifact? flat-LR vs warmup+cosine (SPEC-10, §4.11) =="
echo "   (local sweep, ~6 min on CPU; fixed graph m × {flat-LR, scheduled} × 3 seeds; writes CSV + figure)"
python -m verisim.experiments.horizon_graph_schedule --config configs/horizon_graph_schedule.json \
    --out figures/horizon_graph_schedule.csv --plot figures/horizon_graph_schedule.png

echo "== HS-synth: the PROPOSER-DEPENDENCE synthesis — flat resourcing lift vs graph ceiling in one figure (SPEC-10, §4.9) =="
echo "   (instant; figures-from-records — re-reads horizon_scaling.csv + horizon_graph_scaling.csv, re-runs nothing)"
python -m verisim.experiments.horizon_synthesis \
    --out figures/horizon_synthesis.csv --plot figures/horizon_synthesis.png

echo "== ED1: the DISTRIBUTED prime directive — H_ε(ρ) curve + the tiered-oracle (H17) measurement (SPEC-7, DS6) =="
echo "   (seconds on CPU, no torch; the distributed world's first figure + the oracle-dollar H17 tradeoff)"
python -m verisim.experiments.ed1 --config configs/ed1_dist.json \
    --out figures/ed1_dist.csv --plot figures/ed1_dist.png

echo "== ED1-learned: the distributed prime directive with the REAL learned M_θ (SPEC-7, DS6) =="
echo "   (torch [model] extra; trains the flat DS4 M_θ, then the same tiered loop — the real-model H_ε(ρ) + H17)"
python -m verisim.experiments.ed1_learned --config configs/ed1_learned.json \
    --out figures/ed1_learned.csv --plot figures/ed1_learned.png

echo "== ED2: equal-DOLLAR-budget — does a cheap/escalate tier buy more horizon per \$? (SPEC-7, DS7; H17/H18) =="
echo "   (seconds on CPU, no torch; the horizon-vs-oracle-dollar frontier per tier policy + the competitive ratio)"
python -m verisim.experiments.ed2 --config configs/ed2.json \
    --out figures/ed2.csv --plot figures/ed2.png

echo "== ED2-learned: equal-DOLLAR-budget on the REAL learned M_θ — the honest inverse (SPEC-7, DS7; H17/H18) =="
echo "   (torch [model] extra; the constrained decoder removes gross errors, so only bit-exact buys horizon per \$)"
python -m verisim.experiments.ed2_learned --config configs/ed2_learned.json \
    --out figures/ed2_learned.csv --plot figures/ed2_learned.png

echo "== ED2-smart: the π_c smart-when axis of ED2 — does entropy-gated consultation beat fixed? (SPEC-7, DS7; H9) =="
echo "   (torch [model] extra; the standing H2/H9 null carried in — flat decode entropy is worse than fixed)"
python -m verisim.experiments.ed2_smart --config configs/ed2_smart.json \
    --out figures/ed2_smart.csv --plot figures/ed2_smart.png

echo "== ED3: correction operators — does the distributed world break v0's operator identity? (SPEC-7, DS7; §8.3) =="
echo "   (seconds on CPU, no torch; the partial ReplicasOnlyCorrection breaks the identity on the in-flight medium)"
python -m verisim.experiments.ed3 --config configs/ed3.json \
    --out figures/ed3.csv --plot figures/ed3.png

echo "== ED4-fault: H21 — fault-injected vs fault-free training, evaluated under fault (SPEC-7, DS7) =="
echo "   (torch [model] extra; the DST/BUGGIFY lesson — the data factory's fault_prob dial as the H21 axis)"
python -m verisim.experiments.ed4_fault --config configs/ed4_fault.json \
    --out figures/ed4_fault.csv --plot figures/ed4_fault.png

echo "== ED5: consistency-vs-bit horizon (H19) + the competitive-ratio fit (H18) (SPEC-7, DS8; §9.1, §9.3) =="
echo "   (seconds on CPU, no torch; consistency-faithful outlasts bit-faithful on the in-flight medium; the loop is learning-augmented in the error axis)"
python -m verisim.experiments.ed5 --config configs/ed5.json \
    --out figures/ed5.csv --plot figures/ed5.png

echo "== ED6: distributed counterfactual grounding (H5) — free oracle fault replay vs factual (SPEC-7, DS8; §10.1) =="
echo "   (torch; trains 3 matched-count arms; +counterfactual beats both factual arms on held-out intervention fidelity)"
python -m verisim.experiments.ed6 --config configs/ed6.json \
    --out figures/ed6.csv --plot figures/ed6.png

echo "== ED6 (two-oracle): the consistency oracle is redundant but decision-sufficient + cheaper (SPEC-7, DS8; H12, §10.1) =="
echo "   (seconds on CPU, no torch; subtle in-flight errors are consistency-sufficient (1.0) at ~3.6x lower consult cost; gross durable errors are not (0.0))"
python -m verisim.experiments.ed6_two_oracle --config configs/ed6_two_oracle.json \
    --out figures/ed6_two_oracle.csv --plot figures/ed6_two_oracle.png

echo "== ED4 (consistency level): weaker consistency opens the H19 gap (SPEC-7, DS7; H20/H19, §3.4) =="
echo "   (seconds on CPU, no torch; the gap tracks the in-flight medium — present under eventual, absent under linearizable)"
python -m verisim.experiments.ed4_consistency --config configs/ed4_consistency.json \
    --out figures/ed4_consistency.csv --plot figures/ed4_consistency.png

echo "== SY: the SYSTEM ORACLE — validating the reference oracle against a REAL /bin/sh (SPEC-11) =="
echo "   (seconds on CPU, no torch; macOS-first per SPEC-11 §2.5 — the same cross-POSIX code runs on Linux CI)"
echo "   Gate order is strict: SY3 (safe) -> SY4 (deterministic) -> SY1 (agreement) -> SY2 (debug)."

echo "== SY3: the hermeticity proof — 'no action can occur' negative-control battery (SPEC-11 SO1, H29) =="
python -m verisim.experiments.sy3 --out runs/sy3/records.jsonl \
    --csv figures/sy3_hermeticity.csv --plot figures/sy3_hermeticity.png

echo "== SY4: the determinism attestation — bit-reproducible under the seal (SPEC-11 SO2, H30) =="
python -m verisim.experiments.sy4 --out runs/sy4/records.jsonl --csv figures/sy4_determinism.csv

echo "== SY1: the differential agreement table + the head-to-head H_ε(ρ) overlay (SPEC-11 SO3, H27) =="
echo "   THE FIGURE THAT RETIRES W1: structure-building agreement = 1.000, residual = 0, curve oracle-invariant."
python -m verisim.experiments.sy1 --config configs/sy1.json --out runs/sy1/records.jsonl \
    --csv figures/sy1_agreement.csv --plot figures/sy1_agreement.png

echo "== SY2: the differential debugger — the localized divergence atlas + the teeth control (SPEC-11 SO4, H28) =="
python -m verisim.experiments.sy2 --out runs/sy2/records.jsonl --csv figures/sy2_disagreements.csv

echo "== done: figures/{e1_curve,e2_policies,e3_operators,calibration,e4_ablation,objective,representation,auto_search,en1_curve,en2_policies,en3_operators,en4_graph_vs_flat,en8_grounding,en9_contrastive,en8_scale,en9_scale,en8_capacity,en9_negatives,en7_invariance,en5_selfheal,en6_counterfactual,en8_ls3_hero,en10_two_oracle,eh1_curve,eh1_composition,eh2_policies,eh3_operators,eh4_factored_vs_flat,eh4_drift,eh5_subsystem_policy,eh5_heads,eh_h14_interleaving,eh_h14_scale,eh7_invariance,eh8_privilege,eh6_two_oracle,eh_h13_scale,eh9_denial_weighted,eh6_counterfactual,eh_stream,synthesis_floor_cliff,horizon_scaling,horizon_scaling_xl,horizon_data_scaling,horizon_joint_scaling,horizon_host_scaling,horizon_graph_scaling,horizon_graph_data_scaling,horizon_graph_world_scaling,horizon_graph_joint_scaling,horizon_graph_schedule,horizon_synthesis,ed1_dist,ed1_learned,ed2,ed2_learned,ed2_smart,ed3,ed4_fault,ed4_consistency,ed5,ed6,ed6_two_oracle,sy1_agreement,sy2_disagreements,sy3_hermeticity,sy4_determinism}.{png,csv} =="
