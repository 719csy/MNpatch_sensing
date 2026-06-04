# Figure 2 / Figure 3 Data-Efficiency Target Completion Checklist

Generated: 2026-06-04.

## Requested Matrix

| Figure | Dimension | Devices | Models | N-grid | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Figure 2 | 2D | soft MMP; sheet piezo; ultrasound | SRCNN; U-Net; vanilla diffusion; DPS inverse diffusion; operator-conditioned diffusion | 5 train-N values | 3 | Complete |
| Figure 3 | 3D | soft MMP; sheet piezo; ultrasound | SRCNN 3D proxy; direct scalar regression; 3D U-Net; kernel GP; vanilla 3D diffusion; DPS inverse diffusion; Eapp-only diffusion; operator-conditioned diffusion | 6 train-N values | 3 | Complete |

## Output Files

- `fig2_metrics_by_N.csv`: 225 rows.
- `fig3_metrics_by_N.csv`: 432 rows.
- `fig23_N90_AULC_DER_summary.csv`: 39 rows.
- `model_device_coverage_matrix.csv`: 39 device-model coverage rows.
- `FIG23_DATA_EFFICIENCY_RESULTS_REPORT.md`: summary report.
- `OPENAI_GPT5_NCS_AUDIT_VALIDATED.md`: reviewer-style model audit validated against local CSV values.

## Main Metrics

- `MAE_kPa`: Figure 2 primary reconstruction error.
- `mae_zbottom_mm`: Figure 3 primary depth error.
- `AULC_lower_better`: whole learning-curve area.
- `N90`: self-plateau sample efficiency.
- `N_to_operator_quality`: samples needed to reach the shared operator-conditioned target.
- `DER_common_quality_vs_operator`: common-target data-efficiency ratio.
- `common_quality_reached`: censoring flag for whether a model reached the shared target.

## Evidence Caveat

The requested matrix is complete. Publication-grade claims still require full retrained N-grid deep-learning runs for surrogate rows and uncertainty intervals for the learning-curve metrics.
