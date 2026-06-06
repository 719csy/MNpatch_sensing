# Main Results Summary

## Design

This run implements the page 32-64 task as a reproducible audit and benchmark over existing 2D and 3D simulation outputs. Clinical memory is represented as lesion geometry only. Mechanics, depth, contrast, heterogeneity, and operator variables are handled separately.

## Key Quantitative Findings

- Best held-out geometry distribution match by MMD: `G2_clinical_memory_train`.
- Decoupling gate across randomized depth/stiffness variables: `PASS`.
- 2D operator + clinical mask MAE: 1.728 kPa vs operator-only U-Net 2.388 kPa and vanilla diffusion 6.711 kPa.
- 2D operator + clinical mask Dice: 0.974; Boundary F1: 0.975.
- Best MMP 3D z-bottom method: `D5_operator_plus_clinical_geometry_plus_depth_physics` with MAE_z_bottom 0.201 mm, Cov90 0.936, IntervalIoU 0.851.

PSNR/SSIM are deliberately not used as the primary evidence. The report emphasizes support, boundary, depth posterior, calibration, and OOD/failure behavior.
