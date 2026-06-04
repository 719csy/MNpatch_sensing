# Figure 2 / Figure 3 data-efficiency comparison results

Generated: 2026-06-04 12:55

## What is now covered

- Figure 2 2D apparent rigidity reconstruction: MMP, sheet piezo, ultrasound; SRCNN, U-Net, vanilla diffusion, DPS inverse diffusion, operator-conditioned diffusion.

- Figure 3 3D depth/posterior reconstruction: MMP, sheet piezo, ultrasound; SRCNN proxy, direct scalar regression, 3D U-Net, kernel GP, vanilla 3D diffusion, DPS inverse diffusion, Eapp-only diffusion, operator-conditioned diffusion.

- Outputs include metrics-by-N, self-N90, AULC, common-target DER, censoring flags, plateau flags, and local visualizations.


## Evidence level

- Figure 3 full-N endpoints are anchored to existing `Data/fig3/metrics/depth_metrics.csv` whenever the corresponding operator/model exists.

- Figure 2 MMP endpoints are anchored to existing Figure 2 benchmark tables where the metric definition is compatible.

- Piezo/ultrasound Figure 2 curves and SRCNN 3D are fast surrogate benchmarks from current domain-randomized data, not completed heavy deep-learning N-grid retraining.


## Common-target DER summary

Common target = 10% above the operator-conditioned model's max-N error within each Figure/device panel. `common_quality_reached=False` means the listed DER is a lower-bound censoring estimate.


| figure | device_label | model_label | primary_metric | N90 | AULC_lower_better | common_quality_threshold | N_to_operator_quality | DER_common_quality_vs_operator | common_quality_reached | plateau_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 2 | soft MMP | DPS inverse diffusion | MAE_kPa | 144.000 | 9.148 | 1.900 | 180.000 | 1.250 | False | False |
| Figure 2 | soft MMP | SRCNN | MAE_kPa | 96.000 | 8.992 | 1.900 | 180.000 | 1.250 | False | False |
| Figure 2 | soft MMP | U-Net | MAE_kPa | 96.000 | 8.702 | 1.900 | 180.000 | 1.250 | False | False |
| Figure 2 | soft MMP | operator-conditioned diffusion | MAE_kPa | 144.000 | 3.323 | 1.900 | 144.000 | 1.000 | True | False |
| Figure 2 | soft MMP | vanilla diffusion | MAE_kPa | 144.000 | 7.508 | 1.900 | 180.000 | 1.250 | False | False |
| Figure 2 | sheet piezo | DPS inverse diffusion | MAE_kPa | 144.000 | 10.457 | 8.691 | 48.000 | 1.000 | True | False |
| Figure 2 | sheet piezo | SRCNN | MAE_kPa | 96.000 | 10.859 | 8.691 | 96.000 | 2.000 | True | False |
| Figure 2 | sheet piezo | U-Net | MAE_kPa | 144.000 | 10.811 | 8.691 | 48.000 | 1.000 | True | False |
| Figure 2 | sheet piezo | operator-conditioned diffusion | MAE_kPa | 144.000 | 9.153 | 8.691 | 48.000 | 1.000 | True | False |
| Figure 2 | sheet piezo | vanilla diffusion | MAE_kPa | 24.000 | 62.236 | 8.691 | 180.000 | 3.750 | False | True |
| Figure 2 | ultrasound | DPS inverse diffusion | MAE_kPa | 144.000 | 15.676 | 8.818 | 144.000 | 3.000 | True | False |
| Figure 2 | ultrasound | SRCNN | MAE_kPa | 96.000 | 13.512 | 8.818 | 180.000 | 3.750 | False | False |
| Figure 2 | ultrasound | U-Net | MAE_kPa | 144.000 | 15.780 | 8.818 | 144.000 | 3.000 | True | False |
| Figure 2 | ultrasound | operator-conditioned diffusion | MAE_kPa | 144.000 | 9.280 | 8.818 | 48.000 | 1.000 | True | False |
| Figure 2 | ultrasound | vanilla diffusion | MAE_kPa | 12.000 | 64.645 | 8.818 | 180.000 | 3.750 | False | True |
| Figure 3 | soft MMP | SRCNN 3D proxy | mae_zbottom_mm | 200.000 | 0.646 | 0.221 | 1500.000 | 15.000 | False | True |
| Figure 3 | soft MMP | direct scalar regression | mae_zbottom_mm | 200.000 | 1.397 | 0.221 | 1500.000 | 15.000 | False | True |
| Figure 3 | soft MMP | DPS inverse diffusion | mae_zbottom_mm | 400.000 | 0.446 | 0.221 | 1500.000 | 15.000 | False | True |
| Figure 3 | soft MMP | Eapp-only diffusion | mae_zbottom_mm | 200.000 | 0.507 | 0.221 | 1500.000 | 15.000 | False | True |
| Figure 3 | soft MMP | kernel GP | mae_zbottom_mm | 800.000 | 0.572 | 0.221 | 1500.000 | 15.000 | False | True |
| Figure 3 | soft MMP | operator-conditioned diffusion | mae_zbottom_mm | 400.000 | 0.213 | 0.221 | 100.000 | 1.000 | True | True |
| Figure 3 | soft MMP | 3D U-Net | mae_zbottom_mm | 400.000 | 0.696 | 0.221 | 1500.000 | 15.000 | False | True |
| Figure 3 | soft MMP | vanilla 3D diffusion | mae_zbottom_mm | 50.000 | 0.539 | 0.221 | 1500.000 | 15.000 | False | True |
| Figure 3 | sheet piezo | SRCNN 3D proxy | mae_zbottom_mm | 800.000 | 1.088 | 0.674 | 1500.000 | 30.000 | False | True |
| Figure 3 | sheet piezo | direct scalar regression | mae_zbottom_mm | 800.000 | 1.164 | 0.674 | 1500.000 | 30.000 | False | True |
| Figure 3 | sheet piezo | DPS inverse diffusion | mae_zbottom_mm | 200.000 | 0.465 | 0.674 | 50.000 | 1.000 | True | True |
| Figure 3 | sheet piezo | Eapp-only diffusion | mae_zbottom_mm | 200.000 | 0.774 | 0.674 | 1500.000 | 30.000 | False | True |
| Figure 3 | sheet piezo | kernel GP | mae_zbottom_mm | 400.000 | 0.665 | 0.674 | 200.000 | 4.000 | True | True |
| Figure 3 | sheet piezo | operator-conditioned diffusion | mae_zbottom_mm | 200.000 | 0.623 | 0.674 | 50.000 | 1.000 | True | True |
| Figure 3 | sheet piezo | 3D U-Net | mae_zbottom_mm | 400.000 | 1.113 | 0.674 | 1500.000 | 30.000 | False | True |
| Figure 3 | sheet piezo | vanilla 3D diffusion | mae_zbottom_mm | 100.000 | 0.617 | 0.674 | 50.000 | 1.000 | True | True |
| Figure 3 | ultrasound | SRCNN 3D proxy | mae_zbottom_mm | 800.000 | 1.015 | 0.242 | 1500.000 | 7.500 | False | True |
| Figure 3 | ultrasound | direct scalar regression | mae_zbottom_mm | 800.000 | 1.934 | 0.242 | 1500.000 | 7.500 | False | True |
| Figure 3 | ultrasound | DPS inverse diffusion | mae_zbottom_mm | 400.000 | 0.696 | 0.242 | 1500.000 | 7.500 | False | True |
| Figure 3 | ultrasound | Eapp-only diffusion | mae_zbottom_mm | 800.000 | 0.627 | 0.242 | 1500.000 | 7.500 | False | False |
| Figure 3 | ultrasound | kernel GP | mae_zbottom_mm | 800.000 | 0.848 | 0.242 | 1500.000 | 7.500 | False | False |
| Figure 3 | ultrasound | operator-conditioned diffusion | mae_zbottom_mm | 400.000 | 0.265 | 0.242 | 200.000 | 1.000 | True | True |
| Figure 3 | ultrasound | 3D U-Net | mae_zbottom_mm | 800.000 | 1.102 | 0.242 | 1500.000 | 7.500 | False | False |
| Figure 3 | ultrasound | vanilla 3D diffusion | mae_zbottom_mm | 400.000 | 0.508 | 0.242 | 1500.000 | 7.500 | False | True |


## Main interpretation

- Figure 2: comparison matrix is complete; use rows with `DER_common_quality_vs_operator >= 2` as candidate data-efficiency claims.
- Figure 3: comparison matrix is complete; use rows with `DER_common_quality_vs_operator >= 2` as candidate data-efficiency claims.
- Strong NCS claim still requires replacing fast-surrogate rows with full retrained N-grid runs, especially for Figure 2 piezo/ultrasound and Figure 3 SRCNN proxy.
