# Priority 1 Learning-Curve-by-N Results

Generated: 2026-06-06 11:22

## PDF Priority 1 Requirements

- Figure 2 N-grid: 50, 100, 200, 500, 1000, 2000, 5000, 10000.
- Figure 3 N-grid: 100, 250, 500, 1000, 1600, 3000, 5000.
- At least 3 seeds per N.
- Outputs: `fig2_metrics_by_N.csv`, `fig3_metrics_by_N.csv`, `learning_curve_fits.csv`.
- Metrics: N90, N95, AULC, last-doubling gain, seed SD, DER_MAE, DER_CRPS, DER_calibrated_score.

## Important Provenance Caveat

The requested N-grid exceeds the number of unique available source cases/configurations. Rows with `sampling_mode=bootstrap_with_replacement_effective_N` are effective-N bootstrap/augmentation benchmarks from the available domain-randomized library, not newly acquired independent physical or simulation cases.

## Coverage

- Figure 2 rows: 360; device-model cells: 15; N values: [50, 100, 200, 500, 1000, 2000, 5000, 10000]; seeds: [20260606, 20260607, 20260608].
- Figure 3 rows: 504; device-model cells: 24; N values: [100, 250, 500, 1000, 1600, 3000, 5000]; seeds: [20260606, 20260607, 20260608].

## Gate Update

| gate | figure | status | plateau_pass_operator_conditioned | seed_sd_pass_operator_conditioned | N90_observed_operator_conditioned | note |
| --- | --- | --- | --- | --- | --- | --- |
| G2_learning_plateau_priority1 | Figure 2 | YELLOW | True | False | True | Priority 1 learning curves generated; status uses operator-conditioned model across all devices. |
| G3_method_level_data_efficiency_priority1 | Figure 2 | YELLOW | True | False | True | DER_CRPS>=2 signal is checked together with plateau, seed-SD, and observed-N90 gates before allowing a manuscript-level PASS. |
| G2_learning_plateau_priority1 | Figure 3 | YELLOW | False | False | False | Priority 1 learning curves generated; status uses operator-conditioned model across all devices. |
| G3_method_level_data_efficiency_priority1 | Figure 3 | YELLOW | False | False | False | DER_CRPS>=2 signal is checked together with plateau, seed-SD, and observed-N90 gates before allowing a manuscript-level PASS. |

## Learning-Curve Fits

| figure | device_label | model_label | MAE_N90 | MAE_N95 | MAE_AULC | MAE_last_doubling_gain_fraction | MAE_seed_sd_fraction_at_max_N | DER_MAE | DER_CRPS | DER_calibrated_score | priority1_plateau_pass | priority1_seed_sd_pass | unique_source_cases_ge_N90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 2 | sheet piezo | DPS inverse diffusion | 500 | 500 | 4.304 | -0.01662 | 0.4988 | 0.1 | 0.1 | 0.1 | True | False | False |
| Figure 2 | sheet piezo | SRCNN | 5000 | 1e+04 | 6.094 | 0.07201 | 0.1054 | 1 | 2 | 2 | False | False | False |
| Figure 2 | sheet piezo | U-Net | 500 | 500 | 4.632 | -0.02551 | 0.3968 | 0.1 | 0.04 | 0.1 | True | False | False |
| Figure 2 | sheet piezo | operator-conditioned diffusion | 5000 | 1e+04 | 8.189 | 0.007255 | 0.07471 | 1 | 1 | 1 | True | True | False |
| Figure 2 | sheet piezo | vanilla diffusion | 1e+04 | 1e+04 | 61.98 | 0.02415 | 0.064 | 2 | 2 | 2 | True | True | False |
| Figure 2 | soft MMP | DPS inverse diffusion | 500 | 2000 | 1.865 | 0.05781 | 0.6683 | 0.5 | 1 | 0.5 | False | False | False |
| Figure 2 | soft MMP | SRCNN | 500 | 2000 | 1.9 | 0.1109 | 0.4461 | 0.5 | 2 | 0.5 | False | False | False |
| Figure 2 | soft MMP | U-Net | 1000 | 2000 | 6.359 | 0.1265 | 0.4825 | 1 | 2 | 1 | False | False | False |
| Figure 2 | soft MMP | operator-conditioned diffusion | 1000 | 2000 | 1.954 | -0.02824 | 0.3029 | 1 | 1 | 1 | True | False | False |
| Figure 2 | soft MMP | vanilla diffusion | 50 | 50 | 6.004 | -0.04045 | 0.3704 | 0.05 | 0.1 | 0.05 | True | False | True |
| Figure 2 | ultrasound | DPS inverse diffusion | 100 | 100 | 9.44 | -0.03522 | 0.7516 | 0.2 | 0.2 | 0.2 | True | False | True |
| Figure 2 | ultrasound | SRCNN | 500 | 1000 | 9.155 | -0.03158 | 0.1891 | 1 | 1 | 1 | True | False | False |
| Figure 2 | ultrasound | U-Net | 50 | 50 | 12.29 | -0.09903 | 0.9075 | 0.1 | 0.1 | 0.1 | False | False | True |
| Figure 2 | ultrasound | operator-conditioned diffusion | 500 | 500 | 8.381 | -0.004022 | 0.06893 | 1 | 1 | 1 | True | True | False |
| Figure 2 | ultrasound | vanilla diffusion | 1e+04 | 1e+04 | 65.35 | 0.01745 | 0.08246 | 20 | 20 | 20 | True | True | False |
| Figure 3 | sheet piezo | 3D U-Net | 1600 | 3000 | 1.062 | 0.005872 | 0.0067 | 1.6 | 4 | 4 | True | True | True |
| Figure 3 | sheet piezo | DPS inverse diffusion | 1000 | 1000 | 0.4402 | -0.002696 | 0.03888 | 1 | 4 | 2 | True | True | True |
| Figure 3 | sheet piezo | Eapp-only diffusion | 3000 | 5000 | 0.7585 | 0.006422 | 0.01268 | 3 | 12 | 12 | True | True | False |
| Figure 3 | sheet piezo | SRCNN 3D proxy | 1000 | 3000 | 1.07 | 0.004273 | 0.005093 | 1 | 4 | 4 | True | True | True |
| Figure 3 | sheet piezo | direct scalar regression | 1000 | 3000 | 1.145 | 0.004273 | 0.005093 | 1 | 4 | 4 | True | True | True |
| Figure 3 | sheet piezo | kernel GP | 3000 | 5000 | 0.6501 | 0.01593 | 0.01819 | 3 | 12 | 12 | True | True | False |
| Figure 3 | sheet piezo | operator-conditioned diffusion | 1000 | 1000 | 0.6194 | -0.001603 | 0.01942 | 1 | 1 | 1 | True | True | True |
| Figure 3 | sheet piezo | vanilla 3D diffusion | 1000 | 1000 | 0.6209 | -0.007159 | 0.04147 | 1 | 4 | 4 | True | True | True |
| Figure 3 | soft MMP | 3D U-Net | 1000 | 5000 | 0.6767 | 0.01146 | 0.01204 | 2 | 2 | 2 | True | True | True |
| Figure 3 | soft MMP | DPS inverse diffusion | 1000 | 1600 | 0.4271 | 0.01569 | 0.01467 | 2 | 2 | 2 | True | True | True |
| Figure 3 | soft MMP | Eapp-only diffusion | 1000 | 5000 | 0.4932 | 0.01431 | 0.02036 | 2 | 2 | 2 | True | True | True |
| Figure 3 | soft MMP | SRCNN 3D proxy | 1000 | 1000 | 0.6402 | 0.001019 | 0.01439 | 2 | 2 | 2 | True | True | True |
| Figure 3 | soft MMP | direct scalar regression | 1000 | 1000 | 1.386 | 0.001019 | 0.01439 | 2 | 2 | 2 | True | True | True |
| Figure 3 | soft MMP | kernel GP | 5000 | 5000 | 0.5942 | 0.06792 | 0.02519 | 10 | 10 | 10 | False | True | False |
| Figure 3 | soft MMP | operator-conditioned diffusion | 500 | 3000 | 0.2065 | 0.002427 | 0.05182 | 1 | 1 | 1 | True | True | True |
| Figure 3 | soft MMP | vanilla 3D diffusion | 250 | 250 | 0.5373 | 0.01064 | 0.02895 | 0.5 | 0.5 | 0.5 | True | True | True |
| Figure 3 | ultrasound | 3D U-Net | 3000 | 3000 | 0.9953 | 0.04399 | 0.03184 | 6 | 6 | 6 | True | True | False |
| Figure 3 | ultrasound | DPS inverse diffusion | 1000 | 3000 | 0.5317 | 0.02071 | 0.03487 | 2 | 2 | 2 | True | True | True |
| Figure 3 | ultrasound | Eapp-only diffusion | 1600 | 3000 | 0.5576 | 0.03188 | 0.03657 | 3.2 | 3.2 | 3.2 | True | True | True |
| Figure 3 | ultrasound | SRCNN 3D proxy | 1000 | 1600 | 0.858 | 0.01901 | 0.04673 | 2 | 3.2 | 2 | True | True | True |
| Figure 3 | ultrasound | direct scalar regression | 1000 | 1600 | 1.635 | 0.01901 | 0.04673 | 2 | 3.2 | 2 | True | True | True |
| Figure 3 | ultrasound | kernel GP | 1600 | 3000 | 0.6961 | 0.02657 | 0.01906 | 3.2 | 3.2 | 3.2 | True | True | True |
| Figure 3 | ultrasound | operator-conditioned diffusion | 500 | 500 | 0.2337 | 0.01035 | 0.02223 | 1 | 1 | 1 | True | True | True |
| Figure 3 | ultrasound | vanilla 3D diffusion | 500 | 500 | 0.4993 | 0.001527 | 0.01046 | 1 | 1 | 0.5 | True | True | True |

## Manuscript Use

Use these files to close Priority 1 at the scaffold/effective-N level. For a strong NCS-level claim, replace bootstrap/effective-N rows with independent full N-grid training libraries where possible and keep the censoring/provenance columns visible.