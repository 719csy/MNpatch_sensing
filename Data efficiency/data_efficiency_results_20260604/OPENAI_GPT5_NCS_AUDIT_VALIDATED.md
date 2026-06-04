# GPT-5 NCS Audit, Validated Against Local CSV Results

Source model check: `gpt-5-chat-latest`, OpenAI API, 2026-06-04.

This file preserves the useful reviewer-style assessment from the GPT-5 audit, but corrects model/device-specific claims against the generated local tables:

- `fig2_metrics_by_N.csv`
- `fig3_metrics_by_N.csv`
- `fig23_N90_AULC_DER_summary.csv`
- `model_device_coverage_matrix.csv`

## Verdict

The Figure 2 / Figure 3 data-efficiency matrix is now complete for the requested internal benchmark target, but it is still not sufficient for a final Nature Computational Science quantitative claim unless surrogate rows are replaced by full retrained N-grid experiments and uncertainty intervals are added.

## Coverage

- Figure 2, 2D: 3 devices x 5 models = 15 device-model cells.
- Figure 2 devices: soft MMP, sheet piezo, ultrasound.
- Figure 2 models: SRCNN, U-Net, vanilla diffusion, DPS inverse diffusion, operator-conditioned diffusion.
- Figure 2 grid: 5 train-N values x 3 seeds per cell.
- Figure 3, 3D: 3 devices x 8 models = 24 device-model cells.
- Figure 3 devices: soft MMP, sheet piezo, ultrasound.
- Figure 3 models: SRCNN 3D proxy, direct scalar regression, 3D U-Net, kernel GP, vanilla 3D diffusion, DPS inverse diffusion, Eapp-only diffusion, operator-conditioned diffusion.
- Figure 3 grid: 6 train-N values x 3 seeds per cell.

## Metrics

- Primary Figure 2 metric: `MAE_kPa`, lower is better.
- Primary Figure 3 metric: `mae_zbottom_mm`, lower is better.
- Whole-curve metric: `AULC_lower_better`.
- Self-plateau metric: `N90`.
- Main comparative metric: `DER_common_quality_vs_operator`, defined as N needed by a model divided by N needed by the operator-conditioned model to reach the same 10%-relaxed operator-conditioned max-N target.
- Censoring flag: `common_quality_reached`; if false, DER is a lower-bound estimate because the model did not reach the common target within the N-grid.

## Supported Claims

- The requested comparison matrix is complete for both Figure 2 and Figure 3.
- Figure 2 soft MMP: operator-conditioned diffusion reaches the common target at N=144. Other baselines do not reach it within the grid; their reported DER=1.25 is a censored lower-bound, not a strong multiplier claim.
- Figure 2 sheet piezo: operator-conditioned diffusion reaches target at N=48. SRCNN reaches at N=96, DER=2.0 observed. Vanilla diffusion does not reach target; DER=3.75 is censored.
- Figure 2 ultrasound: operator-conditioned diffusion reaches target at N=48. DPS inverse diffusion and U-Net reach at N=144, DER=3.0 observed. SRCNN and vanilla diffusion do not reach target; DER=3.75 is censored.
- Figure 3 soft MMP: operator-conditioned diffusion reaches target at N=100. All other listed models fail the common target within the grid; DER=15.0 is a censored lower-bound.
- Figure 3 sheet piezo: operator-conditioned diffusion reaches target at N=50. Kernel GP reaches at N=200, DER=4.0 observed. DPS inverse diffusion and vanilla 3D diffusion reach at N=50, DER=1.0. SRCNN 3D proxy, direct scalar regression, Eapp-only diffusion, and 3D U-Net are censored at DER=30.0.
- Figure 3 ultrasound: operator-conditioned diffusion reaches target at N=200. All other listed models fail the common target within the grid; DER=7.5 is a censored lower-bound.

## Required Remaining Upgrades For NCS-Level Claims

- Replace fast-surrogate Figure 2 piezo/ultrasound rows and Figure 3 SRCNN proxy rows with full retrained N-grid benchmarks.
- Add bootstrap or seed-level confidence intervals for N90, AULC, DER, MAE, CRPS, and coverage metrics.
- Add formal statistical tests or paired bootstrap comparisons for AULC and common-target DER.
- Justify the 10% common target threshold or provide sensitivity at 5%, 10%, and 20%.
- Show uncertainty bands on learning curves and mark censored DER values visually.
- Document whether operator-conditioned diffusion uses one shared conditional model across devices or per-device models.

## Use In Manuscript

Use the local programmatic CSV/PNG outputs as quantitative evidence. Use API-generated images only as visual style references unless all text and numeric labels are overlaid from local tables.
