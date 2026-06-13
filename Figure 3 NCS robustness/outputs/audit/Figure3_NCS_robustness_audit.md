# Figure 3 NCS Robustness Audit

## Verdict
- Manuscript robustness gate: `PASS`.
- Reason: minimum high-fidelity, paired benchmark and posterior aggregate gates passed

## High-fidelity domain-randomized MMP library
- Planned cases: `512`.
- Complete cases: `512`.
- Complete strain rows: `1536` / `1536`.
- Observed smoke-case COMSOL time: `1743` s/case; estimated remaining wall time: `0.0` h if run serially.

## Paired device benchmark
- Paired rows: `150` across `15` conditions.
- MMP full-field rows complete: `150` / `150`.
- Piezo full-COMSOL rows complete: `150` / `150`.
- US full-COMSOL rows complete: `150` / `150`.
- All three complete on same paired row: `150` / `150`.

## Missing queues
- MMP exact readout queue: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_ncs_robustness_audit_20260531\fig3_missing_method1_mmp_readout_queue.csv`.
- Piezo full-COMSOL queue: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_ncs_robustness_audit_20260531\fig3_missing_piezo_full_comsol_queue.csv`.
- Piezo runner-ready manifest: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_ncs_robustness_audit_20260531\fig3_missing_piezo_runner_manifest.csv`.
- US full-COMSOL queue: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_ncs_robustness_audit_20260531\fig3_missing_us_full_comsol_queue.csv`.
- US runner-ready manifest: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_ncs_robustness_audit_20260531\fig3_missing_us_runner_manifest.csv`.
- Method2b 512 completion queue: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_ncs_robustness_audit_20260531\fig3_method2b_512_completion_queue.csv`.

## Colab posterior aggregate
- Required aggregate: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\method2b_512_five_seed_ood_full512_20260611\aggregate\method2b_512_five_seed_posterior_aggregate.csv`.
- Required metrics: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\method2b_512_five_seed_ood_full512_20260611\aggregate\method2b_512_five_seed_posterior_metrics.csv`.
- Existing aggregate rows: `7`.
- Command plan: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\method2b_512_five_seed_ood_full512_20260611\method2b_512_five_seed_ood_command_plan.json`.
- Paired statistics: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_posterior_paired_stats_full512_20260611\fig3_posterior_paired_stats.csv`.

## Reviewer gate
- Do not claim Nature Computational Science-level robustness until the full512 five-seed/OOD posterior aggregate exists and paired bootstrap/statistical tests are recomputed from that final aggregate.
