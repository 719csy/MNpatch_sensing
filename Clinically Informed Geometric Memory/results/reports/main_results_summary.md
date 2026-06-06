# Main Results Summary

## Design

This run implements the page 32-64 task as a reproducible audit and benchmark over existing 2D and 3D simulation outputs. Clinical memory is represented as lesion geometry only. Mechanics, depth, contrast, heterogeneity, and operator variables are handled separately.

## Key Quantitative Findings

- Best held-out geometry distribution match by MMD: `G2_clinical_memory_train`.
- Decoupling gate across randomized depth/stiffness variables: `PASS`.
- 2D operator + clinical mask MAE: 1.728 kPa vs operator-only U-Net 2.388 kPa and vanilla diffusion 6.711 kPa.
- 2D operator + clinical mask Dice: 0.974; Boundary F1: 0.975.
- Best MMP 3D z-bottom method: `D5_operator_plus_clinical_geometry_plus_depth_physics` with MAE_z_bottom 0.201 mm, Cov90 0.936, IntervalIoU 0.851.

## Operator Topline Report Card

| operator | MAE_z_bottom | CRPS | Cov90 | Width90 | IntervalIoU | UCE | AUSE | AUROC_fail | failure_positive_count | failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMP | 0.2008 | 0.1297 | 0.9364 | 0.9002 | 0.8511 | 0.08113 | 0.01631 | 1 | 1 | 0.001792 |
| PIEZO | 0.6125 | 0.3331 | 0.9058 | 2.078 | 0.6419 | 0.09379 | 0.03083 | 0.8124 | 106 | 0.19 |
| US | 0.2201 | 0.14 | 0.9385 | 0.9314 | 0.8412 | 0.06485 | 0.01402 | 0.9946 | 1 | 0.001792 |

## Two-Layer Decoupling Audit

Layer 1 is a construction sanity check on randomly recombined geometry and independent mechanics. Layer 2 audits the actual available training/evaluation assets and reports possible leakage or sample-limited uncertainty without overwriting the construction result.

| decoupling_layer | decoupling_status | count |
| --- | --- | --- |
| actual_training_eval_available_2d_samples | WARN_possible_geometry_mechanics_association | 4 |
| actual_training_eval_available_3d_samples | INCONCLUSIVE_sample_limited | 5 |

## PIEZO Operator Caution

The D5 topline is strong for MMP (MAE_z_bottom 0.201 mm) and US (0.220 mm), but weaker for PIEZO (0.612 mm, failure rate 0.190). Do not extrapolate the MMP conclusion to all operators without reporting this PIEZO-specific limitation.

PSNR/SSIM are deliberately not used as the primary evidence. The report emphasizes support, boundary, depth posterior, calibration, and OOD/failure behavior.
