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

echo "== ED6 (two-oracle, learned M_θ): the consistency oracle on the REAL model (SPEC-7, DS8; H12, §10.1) =="
echo "   (torch; the ED2-learned mirror through the other oracle — the constrained decoder's residual errors are subtle/in-flight, so the cheap consistency oracle is decision-sufficient AND cheaper)"
python -m verisim.experiments.ed6_two_oracle_learned --config configs/ed6_two_oracle_learned.json \
    --out figures/ed6_two_oracle_learned.csv --plot figures/ed6_two_oracle_learned.png

echo "== ED4 (consistency level): weaker consistency opens the H19 gap (SPEC-7, DS7; H20/H19, §3.4) =="
echo "   (seconds on CPU, no torch; the gap tracks the in-flight medium — present under eventual, absent under linearizable)"
python -m verisim.experiments.ed4_consistency --config configs/ed4_consistency.json \
    --out figures/ed4_consistency.csv --plot figures/ed4_consistency.png

echo "== ED4 (consistency level, learned M_θ): the ABSOLUTE-predictability H20 on the real model (SPEC-7, DS7; §10.2) =="
echo "   (torch; trains one flat M_θ per level — does linearizable free-run further than eventual? does the H19 gap collapse under strong consistency?)"
python -m verisim.experiments.ed4_consistency_learned --config configs/ed4_consistency_learned.json \
    --out figures/ed4_consistency_learned.csv --plot figures/ed4_consistency_learned.png

echo "== ED DS0/DS3 increment experiments — the deterministic-core anomaly demos (dependency-free, GPU-free) =="
echo "   ED7 Tier-B W1 retirement; ED8 OCC frontier; ED9 write skew; ED10/ED11 Elle; ED12 partial observation;"
echo "   ED13 causal; ED14 quorum; ED15 2PL; ED16 lost update; ED17 dirty read; ED18 message loss (drop);"
echo "   ED19 anti-entropy / read-repair."
python -m verisim.experiments.ed7 --config configs/ed7.json --out figures/ed7.csv --plot figures/ed7.png
python -m verisim.experiments.ed8 --config configs/ed8.json --out figures/ed8.csv --plot figures/ed8.png
python -m verisim.experiments.ed9 --config configs/ed9.json --out figures/ed9.csv --plot figures/ed9.png
python -m verisim.experiments.ed10 --config configs/ed10.json --out figures/ed10.csv --plot figures/ed10.png
python -m verisim.experiments.ed11 --out figures/ed11.csv --plot figures/ed11.png
python -m verisim.experiments.ed12 --config configs/ed12.json --out figures/ed12.csv --plot figures/ed12.png
python -m verisim.experiments.ed13 --config configs/ed13.json --out figures/ed13.csv --plot figures/ed13.png
python -m verisim.experiments.ed14 --config configs/ed14.json --out figures/ed14.csv --plot figures/ed14.png
python -m verisim.experiments.ed15 --config configs/ed15.json --out figures/ed15.csv --plot figures/ed15.png
python -m verisim.experiments.ed16 --config configs/ed16.json --out figures/ed16.csv --plot figures/ed16.png
python -m verisim.experiments.ed17 --config configs/ed17.json --out figures/ed17.csv --plot figures/ed17.png
python -m verisim.experiments.ed18 --config configs/ed18.json --out figures/ed18.csv --plot figures/ed18.png
python -m verisim.experiments.ed19 --config configs/ed19.json --out figures/ed19.csv --plot figures/ed19.png
python -m verisim.experiments.ed20 --config configs/ed20.json --out figures/ed20.csv --plot figures/ed20.png
python -m verisim.experiments.ed21 --config configs/ed21.json --out figures/ed21.csv --plot figures/ed21.png
python -m verisim.experiments.ed22 --config configs/ed22.json --out figures/ed22.csv --plot figures/ed22.png
python -m verisim.experiments.ed23 --config configs/ed23.json --out figures/ed23.csv --plot figures/ed23.png
python -m verisim.experiments.ed24 --config configs/ed24.json --out figures/ed24.csv --plot figures/ed24.png
python -m verisim.experiments.ed25 --config configs/ed25.json --out figures/ed25.csv --plot figures/ed25.png
python -m verisim.experiments.ed26 --config configs/ed26.json --out figures/ed26.csv --plot figures/ed26.png
python -m verisim.experiments.ed27 --config configs/ed27.json --out figures/ed27.csv --plot figures/ed27.png
python -m verisim.experiments.ed28 --config configs/ed28.json --out figures/ed28.csv --plot figures/ed28.png

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

echo "== LP: the LANDMARK GRAPH — the planning altitude above the loop (SPEC-12) =="
echo "   (minutes on CPU; LP1 gates the metric space, LP2 ships the faithful graph, LP3 is the headline)"

echo "== LP1: does the embed() latent encode planning geometry? (SPEC-12 H31) — the metric-space gate =="
echo "   RESULT: H31 refuted (Spearman 0.27 < 0.6) -> build the graph in reachability space (§4 fallback)."
python -m verisim.experiments.lp1 --config configs/lp1.json --out figures/lp1_latent_geometry.csv \
    --plot figures/lp1_latent_geometry.png

echo "== LP2: the faithful landmark graph + the verified-vs-hoped gap (SPEC-12 H32) =="
echo "   RESULT: hoped graph 77% false edges; control-plane verification prunes all (residual 0.000), 0.62x cost."
python -m verisim.experiments.lp2 --config configs/lp2.json --out figures/lp2_faithful_graph.csv \
    --plot figures/lp2_faithful_graph.png

echo "== LP3: goal reach — landmark planning vs flat free-running (SPEC-12 H33) — THE HEADLINE =="
echo "   RESULT: H33 supported. Flat free-running decays with goal-space distance (0.50->0.17, HS3 cliff);"
echo "   landmark planning re-grounding once per hop (rho~0.2) sustains and rises (0.50->0.83) — a 5x"
echo "   far-goal gap that widens with distance and is monotone in the re-grounding budget."
python -m verisim.experiments.lp3 --config configs/lp3.json --out figures/lp3_goal_reach.csv \
    --plot figures/lp3_goal_reach.png

echo "== LP4: edge-metric ablation — reachability vs exact-state edges (SPEC-12 H34) =="
echo "   RESULT: H34 supported. Reachability edges sustain goal reach (0.50->0.83); exact-state edges"
echo "   collapse (~0). Exact-state free-run horizon pinned at 0 (HS3); reachability horizon ~7 (EN10)."
python -m verisim.experiments.lp4 --config configs/lp4.json --out figures/lp4_edge_metric.csv \
    --plot figures/lp4_edge_metric.png

echo "== LP5: landmark placement policy (SPEC-12 H35) =="
echo "   RESULT: H35 split. Belief-variance (uncertainty) buys reach at lower budget (adv +0.10);"
echo "   betweenness underperforms random (adv -0.20). CIs overlap -> leaning supported for uncertainty."
python -m verisim.experiments.lp5 --config configs/lp5.json --out figures/lp5_placement.csv \
    --plot figures/lp5_placement.png

echo "== LP6: replanning policy at equal oracle budget (SPEC-12 H36) =="
echo "   RESULT: H36 supported for the reachability trigger (adv +0.13 over fixed-interval, critical"
echo "   re-ground at B=1); belief-variance trigger miscalibrated (adv -0.20) — the mirror of LP5."
python -m verisim.experiments.lp6 --config configs/lp6.json --out figures/lp6_replanning.csv \
    --plot figures/lp6_replanning.png

echo "== LP7: the LLM-at-the-leaves boundary — search vs myopic walk (SPEC-12 H37) =="
echo "   RESULT: H37 core supported. Search exact at every depth/degree (validity 1.0); myopic walk"
echo "   decays with depth (1.00->0.39) and degree (1.00->0.68). LLM arm deferred (never counted, §9)."
python -m verisim.experiments.lp7 --config configs/lp7.json --out figures/lp7_traversal.csv \
    --plot figures/lp7_traversal.png

echo "== LP8-dist: the cross-world fork — consistency-landmark planning on distsim (SPEC-12 H38) =="
echo "   RESULT: H38 supported in kind. Flat free-running pinned near 0 on the consistency projection"
echo "   (the HS3 analogue); consistency-landmark re-grounding lifts goal reach (adv +0.10, monotone in"
echo "   budget); verified consistency-graph 75% false edges -> 0.000 residual at 0.46x cost. Smaller"
echo "   magnitudes than network LP3 (the dist world is harder) — the honest transfer finding."
python -m verisim.experiments.lp8_dist --config configs/lp8_dist.json \
    --out figures/lp8_dist_goal_reach.csv --plot figures/lp8_dist_goal_reach.png

echo "== LP8-host: the cross-world fork — privilege-landmark planning on hostsim (SPEC-12 H38) =="
echo "   RESULT: H38 supported (more cleanly than dist). Flat free-running decays with distance"
echo "   (0.50->0.00, the HS3 cliff at privilege altitude); privilege-landmark re-grounding sustains"
echo "   (0.50 at far goal), adv +0.18, monotone in budget; verified privilege-graph 74% false edges"
echo "   -> 0.000 residual at 0.25x cost (the cheapest consult of the three worlds)."
python -m verisim.experiments.lp8_host --config configs/lp8_host.json \
    --out figures/lp8_host_goal_reach.csv --plot figures/lp8_host_goal_reach.png

# ---- SPEC-13: speculative world-model rollout (SR) — draft-verify-accept-prefix as the policy ----
echo "== SR2: the accepted-prefix law per world (SPEC-13 H40) — runs first; gates SR1 =="
echo "   RESULT: H40 supported. The accepted prefix grows with g=ε/δ and the empirical mean tracks"
echo "   the i.i.d. law E[a]=α(1-α^k)/(1-α) fed the measured α̂; the split is g (discreteness), not"
echo "   world identity — discrete-regime prefix ~3.7 vs gradual ~11.7, collapsing across worlds."
python -m verisim.experiments.sr2 --config configs/sr2.json \
    --out figures/sr2_accept_law.csv --plot figures/sr2_accept_law.png

echo "== SR1: speculative vs fixed-ρ at equal budget (SPEC-13 H39) — THE HEADLINE (budget crossover) =="
echo "   RESULT: H39 budget-split. Above ρ* speculative reaches full faithfulness (consult-at-break,"
echo "   no wasted clock ticks); below ρ* fixed's uniform spread wins (accept-longest-prefix is"
echo "   budget-greedy — it spends early and free-runs the tail). ρ* ≈ 0.10 (net) / 0.13 (host) / 0.20 (fs)."
python -m verisim.experiments.sr1 --config configs/sr1.json \
    --out figures/sr1_knee.csv --plot figures/sr1_knee.png

echo "== SR3: multi-draft (tree) verification (SPEC-13 H42) =="
echo "   RESULT: H42 supported. best-of-m lifts the accepted prefix ~2.3x under variance (stochastic"
echo "   stalls) but is flat under bias (systematic stalls) — a tree helps iff divergence is variance."
python -m verisim.experiments.sr3 --config configs/sr3.json \
    --out figures/sr3_tree.csv --plot figures/sr3_tree.png

echo "== SR4: calibrated draft length & the EAGLE-2 link (SPEC-13 H41) =="
echo "   RESULT: H41 split. The confidence↔acceptance link transfers (calibration slope +0.22 vs ~0"
echo "   null), but calibrated-k does NOT beat draft-long-everywhere (the oracle-cost inversion §8:"
echo "   verify stops at the break, so a long draft costs no more — calibrating k down only adds calls)."
python -m verisim.experiments.sr4 --config configs/sr4.json \
    --out figures/sr4_calibration.csv --plot figures/sr4_calibration.png

echo "== SR5: the two-tier self-speculative cascade (SPEC-13 H43) =="
echo "   RESULT: H43 refuted (banked negative). A cheap pre-filter does not cut ORACLE calls per"
echo "   faithful step — only the oracle adjudicates, and the cheap tier adds a verify round; use the"
echo "   best drafter directly (cheapness lives on the GPU, free here, not in the oracle)."
python -m verisim.experiments.sr5 --config configs/sr5.json \
    --out figures/sr5_cascade.csv --plot figures/sr5_cascade.png

echo "== SR6: the discreteness law / g-collapse (SPEC-13 H44, deferred fork) =="
echo "   RESULT: H44 partial. The speculative win is hump-shaped in g (small at the K4 cliff, small"
echo "   once free-run is already faithful, peaking in the transition band); worlds share the shape"
echo "   but not exactly the peak (network saturates at lower g) — g governs the shape, collapse approx."
python -m verisim.experiments.sr6 --config configs/sr6.json \
    --out figures/sr6_discreteness.csv --plot figures/sr6_discreteness.png

# ---- SPEC-15: oracle-calibrated conformal consultation (CF) — a guaranteed, explained trigger ----
echo "== CF1: split-conformal coverage gate (H50) + the ρ-vs-coverage frontier (H51) — THE HEADLINE =="
echo "   RESULT: H50 gate holds (undetected ≤ α on exchangeable data) and H51 supported — the calibrated"
echo "   trigger certifies the same α at ~0.43 lower ρ than fixed (a guaranteed RQ2 win: fewer consults"
echo "   to certify the same safety). The oracle is the free, exact, unlimited calibration set."
python -m verisim.experiments.cf1 --config configs/cf1.json \
    --out figures/cf1_coverage_frontier.csv --plot figures/cf1_coverage_frontier.png

echo "== CF4: conformalizability — belief_var vs decode-entropy (H53), the EH2/ED2-smart mechanism =="
echo "   RESULT: H53 supported. Both signals HIT coverage (conformal validity is signal-agnostic), but"
echo "   the calibrated signal saves ~0.50 ρ over fixed (score↔divergence slope +0.13) while the"
echo "   uncalibrated saves ~0 (slope ~0) — conformal *efficiency* is not signal-agnostic."
python -m verisim.experiments.cf4 --config configs/cf4.json \
    --out figures/cf4_signal_split.csv --plot figures/cf4_signal_split.png

echo "== CF2: exchangeability under rollout — static conformal vs ACI (H52), the deepest result =="
echo "   RESULT: H52 supported. Static conformal's undetected rate climbs with depth (0.10 -> 0.41,"
echo "   above α=0.1) — rollout drift breaks exchangeability; ACI, fed the free per-step oracle truth,"
echo "   restores the long-run rate near target (~0.13). Residual deep lag motivates conformal-PID."
python -m verisim.experiments.cf2 --config configs/cf2.json \
    --out figures/cf2_drift_aci.csv --plot figures/cf2_drift_aci.png

echo "== CF3: conformal risk control on the graded undetected-breach loss (H54) =="
echo "   RESULT: H54 supported. Bounding the milder graded loss (vs the 0/1 indicator) buys ~0.22 lower"
echo "   ρ by tolerating near-misses, while certifying E[graded loss] ≤ α."
python -m verisim.experiments.cf3 --config configs/cf3.json \
    --out figures/cf3_risk_control.csv --plot figures/cf3_risk_control.png

echo "== CF5: the conformal cross-world fork — does H50/H51/H53 transfer? (SPEC-15 §7) =="
echo "   RESULT: H50/H51/H53 TRANSFER. The identical torch-free conformal machinery on the host (EH2/H9"
echo "   confirmation) and distributed (ED2-smart challenge) worlds reproduces every result: the gate"
echo "   holds, the calibrated trigger saves +0.42/+0.44/+0.44 ρ vs fixed at α=0.10 (net/host/dist), and"
echo "   the uncalibrated signal saves ~0 everywhere. With breach rate matched (the control), the curves"
echo "   are near-coincident — conformal efficiency is the SIGNAL's, not the WORLD's; the ED2-smart null"
echo "   was the uncalibrated signal, not the distributed world."
python -m verisim.experiments.cf5 --config configs/cf5.json \
    --out figures/cf5_cross_world.csv --plot figures/cf5_cross_world.png

echo "== CF6: the trained-M_theta conformalizability check — is the REAL belief_var calibrated? (SPEC-15) =="
echo "   RESULT: H53 on the real arm. Trains the real structured graph arm and uses its actual RSSM"
echo "   belief_var as the conformal score vs the calibrated/uncalibrated stand-ins (5 seeds). The real"
echo "   network belief_var is NOT calibrated (score<->divergence slope -0.004 ~ 0, vs calibrated +0.10):"
echo "   conformal VALIDITY holds (coverage ~0.10, signal-agnostic) but EFFICIENCY is null (saves -0.015"
echo "   rho, like the uncalibrated stand-in, vs the calibrated +0.45). Efficiency IS conformalizability;"
echo "   the calibrated stand-in is the achievable best case, host belief_var (EH2) the known positive —"
echo "   conformalizability is world/arm-dependent. (Trains 5 graph arms on CPU, ~2-3 min.)"
python -m verisim.experiments.cf6 --config configs/cf6.json \
    --out figures/cf6_real_signal.csv --plot figures/cf6_real_signal.png

# ---- SPEC-18: the product — a frozen faithfulness benchmark + sim-to-emulation ACD environment ----
echo "== PB-bench: the discriminative-validity leaderboard (SPEC-18 H65) — gates the rest =="
echo "   RESULT: H65 supported. The leaderboard stably orders the fidelity ladder (Kendall τ=1.0 across"
echo "   disjoint seed splits at every world) and resolves adjacent tiers above paired seed noise — the"
echo "   benchmark discriminates, so it is worth packaging."
python -m verisim.experiments.pb_bench --config configs/pb_bench.json \
    --out figures/pb_bench_leaderboard.csv --plot figures/pb_bench_leaderboard.png

echo "== PB-transfer: the sim-to-emulation gap vs the real-OS system oracle (SPEC-18 H66/H67) =="
echo "   RESULT: H66 supported — ΔH = H_ε^ref − H_ε^sys ≈ 0 on the validated structure grammar (transfer"
echo "   is essentially lossless, the first such number in faithful-horizon terms; confirms SY1/H27);"
echo "   correction lifts the absolute real-OS horizon with ρ (H67 banked: the bridge is the measurement)."
echo "   skipif-guarded: skipped + not counted when no real shell is present (§2.5)."
python -m verisim.experiments.pb_transfer --config configs/pb_transfer.json \
    --out figures/pb_transfer_gap.csv --plot figures/pb_transfer_gap.png

echo "== PB-transfer-broad: the sim-to-emulation BOUNDARY across grammars (SPEC-18 H66) =="
echo "   RESULT: the boundary mapped. The same ΔH measurement vs the real /bin/sh across grammars of"
echo "   widening scope: ΔH(ρ=0) = 0.000 on the validated 'structural' grammar (lossless, confirms"
echo "   PB-transfer), +0.67 on 'weighted' (cp/mv/rm/chmod/append), +5.75 on 'adversarial' (rm -r/mv/"
echo "   rmdir) — the reference FS model diverges from the real shell exactly on the ops SY1/H27 did not"
echo "   validate. Sharp H67-broad finding: oracle-in-the-loop correction does NOT close the gap (ΔH"
echo "   GROWS with ρ — the loop consults the reference oracle, which can't fix divergence from reality)."
echo "   The first quantified faithfulness boundary (SPEC-3 W1). skipif-guarded (real shell, §2.5)."
python -m verisim.experiments.pb_transfer_broad --config configs/pb_transfer_broad.json \
    --out figures/pb_transfer_broad.csv --plot figures/pb_transfer_broad.png

echo "== PB-pack: contamination control + conformance + metadata (SPEC-18 H68 + milestones) =="
echo "   RESULT: H68 supported — the public-minus-held-out faithful gap separates a public-manifest"
echo "   memorizer (~+0.98) from an honest proposer (~+0.10): the frozen eval is contamination-resistant."
echo "   Conformance 6/6 green (Gymnasium/verifiers contracts); Croissant + datasheet + model-card emitted"
echo "   to bench/. (Trained-arm entries and a real memorizer deferred.)"
python -m verisim.experiments.pb_pack --config configs/pb_pack.json \
    --out figures/pb_pack_contamination.csv --plot figures/pb_pack_contamination.png

# ---- SPEC-17: causal / counterfactual — the oracle as an exact Structural Causal Model ----
echo "== CX0: the oracle is an exact SCM (SPEC-17 H60) — the gate that licenses rung 3 =="
echo "   RESULT: H60 supported. Abduction-action-prediction is bit-exact on every world (rate 1.0):"
echo "   recovering U from the seed reproduces the factual trajectory bit-for-bit, so rung-3"
echo "   counterfactuals are exact and free (abduction is an O(1) reset+replay, not the intractable"
echo "   inference of an oracle-free SCM); the rung-3 trajectory genuinely differs from the factual."
python -m verisim.experiments.cx0 --config configs/cx0.json \
    --out figures/cx0_scm_gate.csv --plot figures/cx0_scm_gate.png

echo "== CX1: the counterfactual effect is hidden-state-dependent (SPEC-17 H61, do-calculus H5) =="
echo "   RESULT: H61 effect-size supported. The distributed world's counterfactual effect amplifies"
echo "   ~3.6x downstream (its persistent partition/crash medium carries the intervention forward),"
echo "   while the on-policy-complete network/host worlds amplify ~1x (the effect washes out) — the"
echo "   do-calculus reading of the mixed H5. The LEARNED lift (CX2-CX4) is deferred to the trained arm."
python -m verisim.experiments.cx1 --config configs/cx1.json \
    --out figures/cx1_counterfactual_effect.csv --plot figures/cx1_counterfactual_effect.png

echo "== CX5: the SCM fork onto the SYSTEM oracles (SPEC-17 H64) — does it survive reality? =="
echo "   RESULT: H64 TRANSFER. Re-running the abduction gate on the real /bin/sh SandboxOracle (fs) and"
echo "   the Tier-B SystemDistOracle (dist) gives abduction-exactness + rung-3 counterfactual-exactness"
echo "   + cf-differs all = 1.0 on both, matching the reference anchor — exact, free rung-3"
echo "   counterfactuals survive the move to the real system. The SY4 DeterminismSeal and the DST seeded"
echo "   scheduler are what make reality an exact SCM (measured, not assumed). A system oracle that is"
echo "   genuinely unavailable (no /bin/sh, no threads) is a disclosed skip, never a pass."
python -m verisim.experiments.cx5 --config configs/cx5.json \
    --out figures/cx5_system_oracle.csv --plot figures/cx5_system_oracle.png

echo "== CX3: the matched-coverage cut — branching vs coverage (SPEC-17 H62) — the open caveat, closed =="
echo "   RESULT: H62 REFUTED. ED6 found a ~2x counterfactual lift on the distributed world but flagged"
echo "   that its counterfactual branches are fault-heavier than the control, so the lift conflates"
echo "   counterfactual BRANCHING with the fault COVERAGE it carries. CX3 (a CPU-scale trained arm + the"
echo "   new causal/coverage.py sampler) matches a factual control and the +counterfactual arm on BOTH"
echo "   example count AND fault-coverage (the _medium statistic), so they differ in branching alone. At"
echo "   matched coverage (0.78) the FACTUAL control strictly beats the counterfactual arm on"
echo "   intervention-exact (0.569 vs 0.426) and medium-recall (0.639 vs 0.480), disjoint CIs both ways."
echo "   So ED6's lift was fault coverage (H21), NOT counterfactual structure — the SPEC-7 §10.1 caveat"
echo "   resolves decisively against branching. (Trains a real distributed M_θ on CPU, ~9 min.)"
python -m verisim.experiments.cx3 --config configs/cx3.json \
    --out figures/cx3_matched_coverage.csv --plot figures/cx3_matched_coverage.png

echo "== CX4: exact-oracle vs learned-model counterfactual augmentation — the CoDA contrast (SPEC-17 H63) =="
echo "   RESULT: H63 SUPPORTED. Operationalizes SPEC §1.1 (a learned model's predictions are unverifiable"
echo "   and drift). On the distributed world, one counterfactual query set is labeled two ways: the exact"
echo "   oracle O(s,a') (valid by construction) and a learned local model M_local (the CoDA stand-in, whose"
echo "   labels inherit its drift). +oracle-aug lifts held-out intervention-exact to 0.394 (over base"
echo "   0.277) while +learned-aug COLLAPSES to 0.064 (BELOW base), disjoint CIs. The mechanism: the"
echo "   learned model's counterfactual samples are only 6% causally valid (oracle 100%), so it injects"
echo "   ~94% invalid data that corrupts training — a learned counterfactual augmenter is not just useless"
echo "   but harmful, exactly where the exact oracle is the unique leverage. (Trains real M_θ, ~13 min.)"
python -m verisim.experiments.cx4 --config configs/cx4.json \
    --out figures/cx4_coda_contrast.csv --plot figures/cx4_coda_contrast.png

# ---- SPEC-16: rollout-stability training — free-oracle DAgger and the exposure-bias cure ----
echo "== RS1: free-oracle DAgger vs teacher forcing on the REAL flat M_θ (SPEC-16 H55) — the headline =="
echo "   RESULT: H55 NOT supported at CPU scale (the pre-registered, first-class negative). Every horizon"
echo "   to date trained teacher-forced and rolled out free-running (the exposure-bias gap HS1.1 caught)."
echo "   DAgger relabels the LEARNER'S OWN drifted states with the oracle (free, exact) and aggregates,"
echo "   but at the affordable CPU scale the flat M_θ is near the H_ε floor with only modest one-step"
echo "   competence: DAgger does NOT lift the free-running horizon over teacher forcing (CIs overlap) —"
echo "   at this scale the gap behaves like fundamental compounding, not a fixable train/deploy mismatch."
echo "   (Trains a real GPT on CPU, ~10 min; whether the cure pays for a competent high-p model at GPU"
echo "   scale is the standing open bet.)"
python -m verisim.experiments.rs1_dagger --config configs/rs1_dagger.json \
    --out figures/rs1_dagger.csv --plot figures/rs1_dagger.png

echo "== RS2: scheduled sampling — the sample_prob tradeoff curve on the REAL structured arm (SPEC-16 H57) =="
echo "   RESULT: H57 NULL. Sweeping max_sample_prob in {0,0.25,0.5,0.75,1.0} on the structured GNN+RSSM"
echo "   arm, neither one-step p (flat ~0.58) nor H_free (best +0.36 at ε=0.3, within seed-CI ±0.70)"
echo "   moves beyond seed noise — no signed bias-stability tradeoff, strengthening NA6's single-point"
echo "   self-forced null into a full curve. (Trains the real graph arm on CPU, ~4 min.)"
python -m verisim.experiments.rs2_scheduled --config configs/rs2_scheduled.json \
    --out figures/rs2_sample_prob_tradeoff.csv --plot figures/rs2_sample_prob_tradeoff.png

echo "== RS3: noise injection — the noise_prob x magnitude grid on the REAL structured arm (SPEC-16 H57) =="
echo "   RESULT: H57 NULL. Sweeping the oracle-relabeled noise lever over noise_prob in {0,0.15,0.3,0.45}"
echo "   x magnitude in {1,2,3} (magnitude = stacked off-trajectory mutations, a knob this adds; mag=1 is"
echo "   byte-identical to every prior caller), NO cell lifts H_free over the no-noise baseline beyond seed"
echo "   noise (best +0.67, within ±1.50) and the p surface is flat (~0.56-0.59). Neither a higher noise"
echo "   rate nor a deeper corruption buys free-running horizon. (Trains the real graph arm on CPU, ~5 min.)"
python -m verisim.experiments.rs3_noise --config configs/rs3_noise.json \
    --out figures/rs3_noise_surface.csv --plot figures/rs3_noise_surface.png

echo "== RS4: the multi-step unrolled loss — the pushforward made exact — on the REAL structured arm (SPEC-16 H55/H57/H58) =="
echo "   RESULT: the nuanced, bankable result that completes the rollout-aware family. RS4 adds the one"
echo "   new trainer (train_unrolled): re-anchor to truth every k steps, unroll the GNN+RSSM on its OWN"
echo "   predictions, and supervise EVERY drifted step with the oracle's exact delta (Brandstetter's"
echo "   pushforward, made exact by the free total oracle). Swept over k in {1,2,4,8}, it is the FIRST"
echo "   rollout-aware lever to move the structured floor — η0 crosses 1 (H_free(ε=0) 1.28->1.60) and"
echo "   raw H_free lifts at the loosest tolerance (k=8 +3.24 at ε=0.5, clearing TF's CI) where RS1 and"
echo "   the RS2/RS3 sweeps all tied teacher forcing — BUT it does not pay net-per-compute"
echo "   (H58): charged 1.5x-4.5x for the extra forwards, net H_free/cost falls with depth. The cure"
echo "   reshapes the error budget, it does not reduce it. (Trains the real graph arm on CPU, ~4 min;"
echo "   k=1 reproduces teacher forcing byte-for-byte, the cost-1.0 anchor.)"
python -m verisim.experiments.rs4_unrolled --config configs/rs4_unrolled.json \
    --out figures/rs4_unroll_depth.csv --plot figures/rs4_unroll_depth.png

echo "== RS6: the net faithful-horizon-per-compute Pareto — the honest verdict (SPEC-16 H58) =="
echo "   RESULT: H58 CONFIRMED at the family level. All four structured-arm trainers (teacher-forced,"
echo "   self-forced DAgger, noise-injected, unrolled) on ONE H_free-vs-total-compute figure, each swept"
echo "   over gradient-step budgets {500,1000,2000}, with a real compute axis (forward passes x params)"
echo "   that CHARGES self-forcing and unrolling for the extra model forwards their data generation costs"
echo "   (teacher forcing and noise pay zero — the oracle makes their data). Teacher forcing is the"
echo "   faithful-horizon-per-compute frontier; no rollout-aware trainer beats it beyond seed noise."
echo "   Rollout-aware training reshapes the error budget, it does not buy horizon per compute on this"
echo "   arm. Also answers RS5 in aggregate. (Trains the real graph arm on CPU, ~10 min.)"
python -m verisim.experiments.rs6_pareto --config configs/rs6_pareto.json \
    --out figures/rs6_net_pareto.csv --plot figures/rs6_net_pareto.png

echo "== RS7: the cross-world fork — rollout-aware trainers on the HOST world (SPEC-16 H59) — the closer =="
echo "   RESULT: H59 CONFIRMED — the verdict TRANSFERS. The four-arm comparison (teacher-forced,"
echo "   self-forced DAgger, noise-injected, unrolled) re-run on the host factored arm (a different"
echo "   oracle over a process/fd/mount grammar; required the new train_host_unrolled). No rollout-aware"
echo "   trainer lifts H_free over teacher forcing beyond seed noise (best +0.44 at e=0.1, within TF's"
echo "   CI half-width +/-2.38; all arms cluster at p~0.19). The rollout-stability picture is a property"
echo "   of the oracle-grounded loop, not one world: the levers reshape the error budget without buying"
echo "   horizon on the host arm too. This completes SPEC-16 (RS1-RS7). (Trains the real host graph arm"
echo "   on CPU, ~12 min.)"
python -m verisim.experiments.rs7_host --config configs/rs7_host.json \
    --out figures/rs7_host_transfer.csv --plot figures/rs7_host_transfer.png

# ---- SPEC-14: neural algorithmic reasoning — diagnosing the structured-arm wall ----
echo "== NA0: does the graph processor execute the reachability propagation? (SPEC-14 H45) — the gate =="
echo "   RESULT: H45 REFUTED. A linear probe of the trained graph arm's per-round node embeddings h_r"
echo "   decodes the oracle's free, exact <=r-hop reachability frontier with a lift 0.119/0.237/0.283 at"
echo "   hops 1/2/3 — ~2-3x the pre-propagation control (h_0 -> F_r), CIs non-overlapping. Message passing"
echo "   DOES execute the multi-hop propagation, so the H_free=0 wall is downstream (decoder/rollout), not"
echo "   the processor — the spec redirects to the decode side and NA1's hint-on-h_r head is redundant."
echo "   (Trains 3 graph arms on CPU, ~1-2 min; NA1-NA4 re-scoped to decoder-side supervision, deferred.)"
python -m verisim.experiments.na0 --config configs/na0.json \
    --out figures/na0_hint_probe.csv --plot figures/na0_hint_probe.png

echo "== NA5: the decode-side rollout diagnostic — is the wall the decoder, not the processor? (SPEC-14) =="
echo "   RESULT: DECODE-SIDE DOMINANT. Free-running the NA0 arm, the frozen in-distribution reachability"
echo "   probe falls with depth (0.87 -> 0.71) but a probe REFIT on the drifted states recovers most of it"
echo "   (0.87 -> 0.83, +0.12 over frozen at the deepest bucket) — the reachability is still in the"
echo "   embedding — while tracks-truth falls ~4x more as state divergence climbs. The processor stays"
echo "   faithful to its own drifted input; the DECODER emits wrong deltas that compound. Confirms NA0's"
echo "   redirection at the rollout level. Pure measurement on the NA0 arm + frozen probe (no trained bet)."
python -m verisim.experiments.na5 --config configs/na5.json \
    --out figures/na5_decode_rollout.csv --plot figures/na5_decode_rollout.png

echo "== NA6: does decoder-side rollout-stability training lift the structured arm's H_free? (SPEC-14) =="
echo "   RESULT: H46-redirected NOT SUPPORTED (the banked compounding negative). Training the structured"
echo "   graph arm teacher-forced vs self-forced (scheduled-sampling DAgger) vs noise-injected, across an"
echo "   epsilon sweep (5 seeds): no decoder-side fix lifts H_free past teacher forcing's seed-CI at any"
echo "   tolerance (best lift +0.92 at eps=0.5, within +-3.04). The H_free=0 wall is exact-tolerance-"
echo "   specific (eps=0 ~floor, eps=0.5 ~26 steps); the fix neither lifts p nor buys horizon — fundamental"
echo "   compounding, not exposure bias, on the structured arm too (sharpening RS1). GPU-scale is the bet."
echo "   (Trains 15 graph arms on CPU, ~4-6 min.)"
python -m verisim.experiments.na6 --config configs/na6.json \
    --out figures/na6_decode_training.csv --plot figures/na6_decode_training.png

echo "== done: figures/{e1_curve,e2_policies,e3_operators,calibration,e4_ablation,objective,representation,auto_search,en1_curve,en2_policies,en3_operators,en4_graph_vs_flat,en8_grounding,en9_contrastive,en8_scale,en9_scale,en8_capacity,en9_negatives,en7_invariance,en5_selfheal,en6_counterfactual,en8_ls3_hero,en10_two_oracle,eh1_curve,eh1_composition,eh2_policies,eh3_operators,eh4_factored_vs_flat,eh4_drift,eh5_subsystem_policy,eh5_heads,eh_h14_interleaving,eh_h14_scale,eh7_invariance,eh8_privilege,eh6_two_oracle,eh_h13_scale,eh9_denial_weighted,eh6_counterfactual,eh_stream,synthesis_floor_cliff,horizon_scaling,horizon_scaling_xl,horizon_data_scaling,horizon_joint_scaling,horizon_host_scaling,horizon_graph_scaling,horizon_graph_data_scaling,horizon_graph_world_scaling,horizon_graph_joint_scaling,horizon_graph_schedule,horizon_synthesis,ed1_dist,ed1_learned,ed2,ed2_learned,ed2_smart,ed3,ed4_fault,ed4_consistency,ed5,ed6,ed6_two_oracle,ed6_two_oracle_learned,ed4_consistency,ed4_consistency_learned,ed7,ed8,ed9,ed10,ed11,ed12,ed13,ed14,ed15,ed16,ed17,ed18,ed19,ed20,ed21,ed22,ed23,ed24,ed25,ed26,ed27,ed28,sy1_agreement,sy2_disagreements,sy3_hermeticity,sy4_determinism,lp1_latent_geometry,lp2_faithful_graph,lp3_goal_reach,lp4_edge_metric,lp5_placement,lp6_replanning,lp7_traversal,lp8_dist_goal_reach,lp8_host_goal_reach,sr1_knee,sr2_accept_law,sr3_tree,sr4_calibration,sr5_cascade,sr6_discreteness,cf1_coverage_frontier,cf2_drift_aci,cf3_risk_control,cf4_signal_split,cf5_cross_world,cf6_real_signal,pb_bench_leaderboard,pb_transfer_gap,pb_transfer_broad,pb_pack_contamination,cx0_scm_gate,cx1_counterfactual_effect,cx5_system_oracle,cx3_matched_coverage,cx4_coda_contrast,rs1_dagger,rs2_sample_prob_tradeoff,rs3_noise_surface,rs4_unroll_depth,rs6_net_pareto,rs7_host_transfer,na0_hint_probe,na5_decode_rollout,na6_decode_training}.{png,csv} =="
