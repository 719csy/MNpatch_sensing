# Figure 2-3 Data Efficiency Experiment Design

This report is based on `data efficiency.pdf` pages 25-68 and the local reference data folders. It is an experiment design and readiness plan, not a claim that the final model has already passed all gates.

## Core Decision

Use two linked but separate data-efficiency experiments:

1. Figure 2: 2D apparent mechanics, `E_app(x,y)`, focused on map, boundary, contrast, posterior calibration, and real-MMP support coverage.
2. Figure 3: 3D latent mechanics, `E(x,y,z)` / `z_bottom`, focused on continuous depth posterior, operator observability, calibration/sharpness, multimodality, and OOD/failure utility.

Do not frame domain randomization as 'large simulated N'. The gate is whether real or held-out readouts fall inside the simulated readout-support manifold and whether learning curves are saturated.

## Existing Reference Coverage

- MMP/MN manifest rows: 2380; QC trainable cases: 445; method1 2D cases: 414; method2 3D cases: 31.
- 3D rigidity mapping rows: 298; depth levels: 14 ([0.3, 0.6, 0.9, 1.0, 1.2, 2.0, 2.4, 3.0, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0]); strain levels: 10.
- Piezo benchmark: 150 readout cases across 15 conditions; full COMSOL rows: 150.
- Ultrasound benchmark: 150 readout cases across 15 conditions; full COMSOL rows: 150.
- Clinical ontology support: 4 datasets, 21 canonical terms, total n=5728.

Important caution: piezo and ultrasound are simulation benchmarks here unless paired real piezo/US files are added. Real-support claims should be anchored on MMP phantom/animal/ex vivo readouts.

## Recommended Category Counts

- Figure 2 should include 4 category axes and 12 named classes: operator/readout (3), morphology/source (3), contrast (3), noise/contact/calibration (3).
- Figure 3 should include 5 category axes and 16 named classes: operator/readout (3), depth (4), thickness (3), contrast (3), contact/load/noise (3).
- For a strong domain-randomization claim, keep each primary ID stratum represented by at least 3 random seeds and split by `config_id`, not by readout row.

## Learning-Curve Rule

Fit `m(N)=m_inf+a*N^-alpha` for lower-is-better metrics and `m(N)=m_inf-a*N^-alpha` for higher-is-better metrics. Report N90, N95, AULC, marginal gain after last doubling, seed SD, and DER = N90(baseline) / N90(ours).

## Gate Summary

A figure is ready only if inventory/leakage, simulator support, learning plateau, method-level data efficiency, calibration/sharpness, and OOD/failure utility are at least PASS/YELLOW with no FAIL. Figure 3 additionally requires a continuous depth posterior gate; merging Figure 2 and Figure 3 additionally requires projection consistency.

## Output Files

- `fig23_existing_data_inventory.csv`: source availability and high-level counts.
- `fig23_reference_category_counts.csv`: observed categories/counts from local manifests.
- `fig23_experiment_design.csv`: recommended Figure 2/3 category plan.
- `fig23_recommended_class_budget.csv`: direct per-class current support and target sample budget.
- `fig23_learning_curve_plan.csv`: N grids, models, metrics.
- `fig23_metric_gate_table.csv`: PASS/YELLOW/FAIL thresholds.
- `fig23_domain_randomization_matrix.csv`: domain randomization dimensions and minimum class counts.
- `fig23_counts_from_reference_data.json`: compact JSON coverage summary.