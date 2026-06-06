# Clinical Memory Claim Check

| Question | Decision | Support |
|---|---:|---|
| Does clinical geometry prior match real lesion shape distribution? | PASS | geometry_distribution_matching.csv; figure_2_geometry_memory |
| Is geometry statistically independent from depth/stiffness? | PASS | independence_tests.csv; figure_2_geometry_memory |
| Does clinical memory improve geometry recovery? | PASS | main_ablation_table.csv; figure_3_2d_ablation |
| Does clinical memory improve mechanics posterior after sensor conditioning? | PASS | main_ablation_table.csv; figure_4_3d_depth_posterior |
| Does geometry-only model fail to infer mechanical variables? | PASS | independence_tests.csv |
| Does OOD uncertainty increase when depth leaves training support? | PASS | ood_failure_metrics.csv; figure_5_uncertainty_ood |

Geometry memory is statistically decoupled from randomized mechanical variables; depth and stiffness cannot be inferred from geometry alone.
