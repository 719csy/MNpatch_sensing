# NCS data-efficiency audit for Figure 2 and Figure 3

Generated: 2026-06-04 11:35

## Executive decision

- Current evidence does **not yet** pass all gates for a strong Nature Computational Science data-efficiency/sim-to-real claim.

- It is suitable for a simulation-grounded Figure 2/3 benchmark, with missing gates explicitly labeled.

- Figure 2 currently uses 4 domain-randomization axes and 15 axis-level classes.

- Figure 3 currently uses 6 domain-randomization axes and 20 axis-level classes.

- Class counts are per axis; some Figure 3 stressor classes are intentionally non-exclusive.


## Gate table

| gate | figure | status | current_value | ncs_threshold | required_action | claim_level |
| --- | --- | --- | --- | --- | --- | --- |
| G0 provenance/leakage | Figure 2 + Figure 3 | YELLOW | duplicate_hash=209; strain_monotonicity_fail=38; figure3_visual_guardrail=True | No config/latent leakage across splits; no truth-derived posterior; visual data plots from real data tables. | Quarantine duplicate-hash and monotonicity-fail cases; add config_id split audit for all learning curves. | Simulation scaffold until split audit is attached. |
| G1 simulator support | Figure 2 | MISSING | real MMP phantom/animal/ex vivo anchors not found | >=90% real/phantom readouts within 95% synthetic support envelope. | Ingest >=20 physical MMP phantom anchors plus all available animal/ex vivo readouts; compute kNN/Mahalanobis/MMD support. | Cannot claim strong sim-to-real domain randomization yet. |
| G1 simulator support | Figure 3 | MISSING | no real MMP support-envelope table found for 3D posterior readouts | >=90% real/phantom 3D readouts within 95% synthetic readout manifold. | Create support_coverage_real_mmp.csv using readout embeddings and matched 3D simulator features. | Simulation benchmark can remain; real-transfer claim must wait. |
| G2 learning plateau | Figure 2 | MISSING | no fig2_metrics_by_N.csv / fitted N90 table found | Last-doubling improvement <3-5%; N90/N95/AULC reported across >=3 seeds. | Run N grid for 2D reconstruction: 50,100,200,500,1k,2k,5k,10k with config-level splits. | Full-data benchmark only, not data-efficiency claim. |
| G2 learning plateau | Figure 3 | MISSING | no fig3_metrics_by_N.csv / fitted N90 table found | Last-doubling improvement <3-5%; N90/N95/AULC reported across >=3 seeds. | Run N grid for 3D posterior: 100,250,500,1k,1.6k current, then 3k/5k if plateau is not reached. | Posterior benchmark only, not data-efficiency claim. |
| G3 method-level data efficiency | Figure 2 + Figure 3 | MISSING | DER_MAE / DER_CRPS not estimable without N90 curves | DER >=2 versus strongest baseline or clearly better CRPS/calibration/failure at same N. | Fit learning curves for U-Net, vanilla diffusion, DPS/kernel GP, and op-conditioned diffusion. | Do not say 'twofold fewer simulations' yet. |
| G4 Figure 2 reconstruction | Figure 2 | YELLOW | MAE=1.728 kPa; SSIM=0.899; Dice=0.974 | MAE/SSIM/Boundary-F1 plateau; uncertainty rejection reduces MAE/CRPS >=25%; posterior maps exported. | Export case-level final posterior mean/std maps and add uncertainty-risk curve. | Reconstruction evidence is strong but not a complete data-efficiency package. |
| G5 Figure 3 posterior | Figure 3 | PASS | MMP: MAE=0.201mm Cov90=0.936 IoU=0.851; PIEZO: MAE=0.612mm Cov90=0.906 IoU=0.642; US: MAE=0.220mm Cov90=0.938 IoU=0.841 | In-support z_bottom MAE <=0.5-0.75mm, interval IoU >=0.55-0.60, calibrated posterior. | Keep as simulation posterior benchmark; attach learning curves and support coverage for main data-efficiency claim. | Passes simulation benchmark; not sufficient alone for NCS data-efficiency claim. |
| G6 calibration/sharpness | Figure 3 | PASS | MMP: Cov90=0.936, Width90=0.900mm; PIEZO: Cov90=0.906, Width90=2.078mm; US: Cov90=0.938, Width90=0.931mm | /Cov90-0.90/ <=0.03 ID or <=0.05 hard/OOD; width not inflated. | If any operator exceeds tolerance, calibrate intervals on validation split and report pre/post calibration. | Use exact operator-specific language. |
| G7 OOD/failure utility | Figure 3 | PASS | MMP: AUROC_fail_1mm=1.000; PIEZO: AUROC_fail_1mm=0.812; US: AUROC_fail_1mm=0.995 | AUROC_fail >=0.75-0.80; risk at retained 80% improves; AUSE improves >=15-20%. | Add failure prevalence/AUPRC and retained-risk curves by depth/OOD group. | May support failure-screening claim if prevalence diagnostics are attached. |
| G8 projection consistency | Figure 2 + Figure 3 | MISSING | projection arrays exist, but no projection_consistency_metrics.csv found | NMAE(Eapp2D, Pz[E3D]) <=0.10-0.15; SSIM >=0.80-0.85; centroid shift <=0.5-1.0mm. | Compute paired 2D/3D projection QC before merging Figure 2 and Figure 3 story. | Do not merge as a single posterior framework until this gate is measured. |


## Class summary

| figure | axis | n_classes | min_current_n | max_current_n | overall_status | class_labels |
| --- | --- | --- | --- | --- | --- | --- |
| Figure 2 | contrast | 3 | 31.0 | 90.0 | YELLOW | matched/low contrast (<=2x); moderate contrast (2-8x); high contrast (>8x) |
| Figure 2 | morphology phenotype | 6 | 5.0 | 74.0 | FAIL | circular/compact; heterogeneous; irregular boundary; low contrast; multifocal; other lesion prior |
| Figure 2 | operator/readout | 3 | 150.0 | 445.0 | YELLOW | MMP soft magnetoelastic microneedle patch; sheet piezoelectric device; ultrasound strain device |
| Figure 2 | source/support tier | 3 | 0.0 | 5728.0 | MISSING | clinical/mask-derived lesion priors; clinical ontology labels; real MMP phantom/animal/ex vivo anchors |
| Figure 3 | contact/load/noise | 3 | 197.0 | 1110.0 | YELLOW | baseline; hard shift; mild shift |
| Figure 3 | contrast | 3 | 235.0 | 970.0 | YELLOW | low/matched contrast (<=2x); moderate contrast (2-5x); high contrast (>5x) |
| Figure 3 | depth | 4 | 400.0 | 400.0 | PASS | deep; mid; ood_deep; shallow |
| Figure 3 | morphological complexity | 4 | 172.0 | 414.0 | FAIL | irregular boundary; heterogeneous core; rim stiffness; multifocality |
| Figure 3 | operator/readout | 3 | 181.0 | 181.0 | PASS | MMP; PIEZO; US |
| Figure 3 | thickness | 3 | 202.0 | 812.0 | FAIL | thin (<1.5 mm); medium (1.5-3.0 mm); thick (>3.0 mm) |


## Primary metric snapshot

| figure | metric_family | primary_method | MAE_kPa | SSIM | Dice | Boundary_F1 | relative_MAE_gain_vs_UNet | interpretation | MAE_zbottom_mm | P90AE_zbottom_mm | CRPS_zbottom_mm | Cov90 | Width90_mm | Interval_IoU | AUROC_fail_1mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 2 | 2D reconstruction | Op-cond. diffusion + mask | 1.727557681491713 | 0.8991818466764006 | 0.973979883243025 | 0.9752988047808764 | 0.27667595840487913 | Strong point reconstruction, but data-efficiency learning curves and real MMP support are still missing. |  |  |  |  |  |  |  |
| Figure 3 | 3D posterior MMP | op_conditioned_diffusion_ours |  |  |  |  |  | Simulation posterior benchmark is strong; final data-efficiency claim still needs learning-curve-by-N and support envelope. | 0.2007922127225366 | 0.2016526548562547 | 0.129654944036262 | 0.9364214639621145 | 0.9002349406792978 | 0.8510702734814407 | 1.0 |
| Figure 3 | 3D posterior PIEZO | op_conditioned_diffusion_ours |  |  |  |  |  | Simulation posterior benchmark is strong; final data-efficiency claim still needs learning-curve-by-N and support envelope. | 0.6124657447371916 | 0.6142995263799222 | 0.333114080261834 | 0.9058306643672496 | 2.077893880213601 | 0.6418650786591994 | 0.8123646967720154 |
| Figure 3 | 3D posterior US | op_conditioned_diffusion_ours |  |  |  |  |  | Simulation posterior benchmark is strong; final data-efficiency claim still needs learning-curve-by-N and support envelope. | 0.2201410517425258 | 0.2209838185043056 | 0.1400344154073463 | 0.9384539842873176 | 0.9313604382826636 | 0.8412092252123843 | 0.9945945945945946 |


## Immediate improvement queue

| priority | figure | gap | action | target_metric | target_threshold | output_file | automation_level |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Figure 2 + Figure 3 | No actual learning-curve-by-N files | Run stratified N-grid training/evaluation with >=3 seeds and config_id-level splits. | N90, N95, AULC, last-doubling gain, seed SD | last-doubling gain <3-5%; DER >=2 if claimed | fig2_metrics_by_N.csv; fig3_metrics_by_N.csv; learning_curve_fits.csv | requires training jobs |
| 2 | Figure 2 + Figure 3 | Real MMP support envelope missing | Ingest physical phantom/animal/ex vivo readouts and compute readout-feature support coverage against synthetic manifold. | support_score, kNN distance, Mahalanobis distance, MMD, conformal p-value | >=90% real/phantom readouts inside 95% synthetic support envelope | support_coverage_real_mmp.csv | requires raw real data |
| 3 | Figure 2 | Publication-ready posterior maps missing | Export case-level posterior mean/std maps and uncertainty rejection curves for selected lesion cases. | MAE, SSIM, Dice, Boundary-F1, uncertainty-retained risk | uncertainty rejection reduces MAE/CRPS >=25% | fig2_case_level_posteriors.csv; fig2_uncertainty_risk.csv | requires inference/export |
| 4 | Figure 3 | Projection consistency before Figure 2/3 merge missing | Project 3D posterior mean to 2D apparent stiffness and compare with Figure 2 Eapp maps. | projection NMAE, SSIM, centroid shift, area difference | NMAE <=0.10-0.15; SSIM >=0.80-0.85; centroid shift <=1mm | projection_consistency_metrics.csv | can be scripted after paired cases are defined |
| 5 | Figure 3 | Calibration tolerance should be operator-specific | Report pre/post conformal or validation calibration without test leakage. | Cov90, Width90, UCE, ENCE, PIT shape | /Cov90-0.90/ <=0.03 ID or <=0.05 hard/OOD | calibration_prepost_metrics.csv | can be scripted from validation posterior samples |
| 6 | Figure 2 + Figure 3 | Duplicate-hash and monotonicity-fail MMP cases in source QC | Quarantine failed cases and enforce split by latent config_id before any data-efficiency training. | duplicate_hash_cases=0 in train/test overlap; monotonicity_fail excluded or labeled | zero leakage; all exclusions logged | split_leakage_audit.csv; qc_exclusion_manifest.csv | can be scripted |


## Generated visualizations

- scorecard: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\visualizations\ncs_readiness_scorecard.png`
- coverage: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\visualizations\domain_randomization_class_coverage.png`
- dashboard: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\visualizations\figure2_figure3_data_efficiency_dashboard.png`


## Interpretation for manuscript claims

- Strong wording allowed now: Figure 3 has strong simulation posterior benchmark metrics for the op-conditioned diffusion model; Figure 2 has strong point reconstruction evidence on the available benchmark table.
- Strong wording not allowed yet: 'domain randomization achieved sim-to-real transfer' or 'twofold data-efficient' until learning curves and real support coverage pass.
- Required NCS-facing metrics: N90/N95/AULC, data-efficiency ratio, support coverage, Cov90/Width90/UCE/ENCE, AUROC/AUPRC failure utility, and 2D/3D projection consistency.

## API visualization run status

- OpenAI image generation succeeded with `gpt-image-1.5`.
- NanoBanana/Gemini image generation succeeded with `gemini-2.5-flash-image`.
- OpenAI output: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\api_generated_visualizations\openai_gpt-image-1.5_ncs_data_efficiency.png`
- NanoBanana/Gemini output: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\api_generated_visualizations\nanobanana_gemini-2.5-flash-image_ncs_data_efficiency.png`
- Visual QC: API-generated figures should be treated as style/concept visualizations because image models can distort table text and numeric labels. The local programmatic PNGs remain the authoritative data-grounded figures for manuscript use.
- Detailed API status: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\API_GENERATION_STATUS.md`
- Detailed API log: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\api_generated_visualizations\api_generation_log.json`
