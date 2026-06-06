# PIEZO Failure Note

The MMP result should not be generalized to all sensing operators. For the D5/op-conditioned clinical geometry + depth physics model, PIEZO has weaker z-bottom performance than MMP and US.

## D5 Topline

| operator | MAE_z_bottom | CRPS | Cov90 | Width90 | IntervalIoU | UCE | AUSE | AUROC_fail | failure_positive_count | failure_rate | failure_threshold_mm | n_eval_records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMP | 0.2008 | 0.1297 | 0.9364 | 0.9002 | 0.8511 | 0.08113 | 0.01631 | 1 | 1 | 0.001792 | 1 | 558 |
| PIEZO | 0.6125 | 0.3331 | 0.9058 | 2.078 | 0.6419 | 0.09379 | 0.03083 | 0.8124 | 106 | 0.19 | 1 | 558 |
| US | 0.2201 | 0.14 | 0.9385 | 0.9314 | 0.8412 | 0.06485 | 0.01402 | 0.9946 | 1 | 0.001792 | 1 | 558 |

## PIEZO Method Ranking By z-bottom MAE

| model_name | MAE_z_bottom | CRPS | Cov90 | Width90 | IntervalIoU | UCE | AUSE | AUROC_fail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D4_operator_plus_clinical_geometry | 0.4328 | 0.3331 | 0.8997 | 2.192 | 0.7348 | 0.3397 | 0.1198 | 0.9959 |
| D5_depth_aware_kernel_gp | 0.605 | 0.4477 | 0.9004 | 2.696 | 0.6608 | 0.3861 | 0.1363 | 0.9588 |
| D5_operator_plus_clinical_geometry_plus_depth_physics | 0.6125 | 0.3331 | 0.9058 | 2.078 | 0.6419 | 0.09379 | 0.03083 | 0.8124 |
| D0_D1_vanilla_3d_diffusion | 0.6173 | 0.445 | 0.8939 | 2.244 | 0.6671 | 0.3143 | 0.0601 | 0.9651 |
| D1_eapp_only_diffusion | 0.7331 | 0.5366 | 0.8664 | 2.213 | 0.6307 | 0.3397 | 0.05431 | 0.9413 |
| D1_operator_only_3d_unet | 1.019 | 1.009 | 0.6387 | 3.8 | 0.4885 | 0.8994 | 0.123 | 0.8212 |
| D1_operator_only_scalar | 1.09 | 0.7763 | 0.7583 | 4.266 | 0.4352 | 0.7488 | 0.4507 | 0.3437 |

## Trend/Cancellation Check

| depth_group | beta_extrap | range_ratio | spearman_rho | pred_vs_true_slope | toward_mean_collapse_flag |
| --- | --- | --- | --- | --- | --- |
| deep |  | 1.389 | 0.9147 | 1.034 | False |
| mid |  | 1.419 | 0.9596 | 1.015 | False |
| ood_deep | 0.06887 | 1.104 | 0.9498 | 0.7931 | False |
| shallow | -0.07029 | 1.336 | 0.9436 | 1.027 | False |

Toward-mean collapse flags for PIEZO D5: none under configured rule.

Manuscript wording should state that clinical memory and depth-aware conditioning are strongest for MMP/US in this benchmark, while PIEZO requires operator-specific qualification and likely additional calibration or model selection.
