# Figure 3 Paired Posterior Statistics

- Metrics source: `Figure 3 NCS robustness\outputs\posterior_aggregate\method2b_512_five_seed_posterior_metrics.csv`.
- Ours models tested: `residual_likelihood_guided_diffusion, prior_init_diffusion_unet, explicit_residual_target_diffusion`.
- Effect size is `baseline_minus_ours`; positive values favor ours for error metrics.

## Top Comparisons
- `ause_abs_error` `srcnn_bz_single_channel` minus `residual_likelihood_guided_diffusion`: mean `0.07155`, 95% CI `[0.06918, 0.07418]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ause_abs_error` `srcnn_bz_single_channel` minus `prior_init_diffusion_unet`: mean `0.07155`, 95% CI `[0.06916, 0.07418]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ause_abs_error` `srcnn_bz_single_channel` minus `explicit_residual_target_diffusion`: mean `0.06683`, 95% CI `[0.06458, 0.06932]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ause_abs_error` `unet_canonical_bxyz` minus `residual_likelihood_guided_diffusion`: mean `0.04616`, 95% CI `[0.0445, 0.0479]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ause_abs_error` `unet_canonical_bxyz` minus `prior_init_diffusion_unet`: mean `0.04615`, 95% CI `[0.04448, 0.04786]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ause_abs_error` `unet_canonical_bxyz` minus `explicit_residual_target_diffusion`: mean `0.04144`, 95% CI `[0.03945, 0.0433]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ause_abs_error` `explicit_residual_target_diffusion` minus `residual_likelihood_guided_diffusion`: mean `0.004729`, 95% CI `[0.003751, 0.005767]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ause_abs_error` `explicit_residual_target_diffusion` minus `prior_init_diffusion_unet`: mean `0.00472`, 95% CI `[0.003726, 0.00572]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ause_abs_error` `prior_init_diffusion_unet` minus `residual_likelihood_guided_diffusion`: mean `9.594e-06`, 95% CI `[-4.948e-05, 6.974e-05]`, Wilcoxon FDR `0.701`, n `25`.
- `ause_abs_error` `residual_likelihood_guided_diffusion` minus `prior_init_diffusion_unet`: mean `-9.594e-06`, 95% CI `[-6.818e-05, 5.176e-05]`, Wilcoxon FDR `0.701`, n `25`.
- `ause_abs_error` `prior_init_diffusion_unet` minus `explicit_residual_target_diffusion`: mean `-0.00472`, 95% CI `[-0.005728, -0.003717]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ause_abs_error` `residual_likelihood_guided_diffusion` minus `explicit_residual_target_diffusion`: mean `-0.004729`, 95% CI `[-0.005742, -0.003743]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `crps_norm` `srcnn_bz_single_channel` minus `explicit_residual_target_diffusion`: mean `0.057`, 95% CI `[0.05297, 0.06063]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `crps_norm` `srcnn_bz_single_channel` minus `residual_likelihood_guided_diffusion`: mean `0.04363`, 95% CI `[0.04104, 0.04608]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `crps_norm` `srcnn_bz_single_channel` minus `prior_init_diffusion_unet`: mean `0.04363`, 95% CI `[0.04094, 0.0461]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `crps_norm` `unet_canonical_bxyz` minus `explicit_residual_target_diffusion`: mean `0.01606`, 95% CI `[0.01276, 0.01861]`, Wilcoxon FDR `6.49e-07`, n `25`.
- `crps_norm` `prior_init_diffusion_unet` minus `explicit_residual_target_diffusion`: mean `0.01337`, 95% CI `[0.01004, 0.01586]`, Wilcoxon FDR `3.59e-06`, n `25`.
- `crps_norm` `residual_likelihood_guided_diffusion` minus `explicit_residual_target_diffusion`: mean `0.01337`, 95% CI `[0.01007, 0.01588]`, Wilcoxon FDR `3.59e-06`, n `25`.
- `crps_norm` `unet_canonical_bxyz` minus `residual_likelihood_guided_diffusion`: mean `0.002688`, 95% CI `[0.002429, 0.002956]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `crps_norm` `unet_canonical_bxyz` minus `prior_init_diffusion_unet`: mean `0.002687`, 95% CI `[0.002433, 0.002954]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `crps_norm` `prior_init_diffusion_unet` minus `residual_likelihood_guided_diffusion`: mean `9.495e-07`, 95% CI `[-9.88e-06, 1.126e-05]`, Wilcoxon FDR `0.571`, n `25`.
- `crps_norm` `residual_likelihood_guided_diffusion` minus `prior_init_diffusion_unet`: mean `-9.495e-07`, 95% CI `[-1.144e-05, 9.972e-06]`, Wilcoxon FDR `0.571`, n `25`.
- `crps_norm` `explicit_residual_target_diffusion` minus `residual_likelihood_guided_diffusion`: mean `-0.01337`, 95% CI `[-0.01585, -0.01011]`, Wilcoxon FDR `3.59e-06`, n `25`.
- `crps_norm` `explicit_residual_target_diffusion` minus `prior_init_diffusion_unet`: mean `-0.01337`, 95% CI `[-0.01585, -0.01015]`, Wilcoxon FDR `3.59e-06`, n `25`.
- `ence` `residual_likelihood_guided_diffusion` minus `explicit_residual_target_diffusion`: mean `11.9`, 95% CI `[11.14, 12.75]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ence` `prior_init_diffusion_unet` minus `explicit_residual_target_diffusion`: mean `11.89`, 95% CI `[11.12, 12.71]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ence` `residual_likelihood_guided_diffusion` minus `prior_init_diffusion_unet`: mean `0.003736`, 95% CI `[-0.01726, 0.02767]`, Wilcoxon FDR `1`, n `25`.
- `ence` `prior_init_diffusion_unet` minus `residual_likelihood_guided_diffusion`: mean `-0.003736`, 95% CI `[-0.02757, 0.01736]`, Wilcoxon FDR `1`, n `25`.
- `ence` `explicit_residual_target_diffusion` minus `prior_init_diffusion_unet`: mean `-11.89`, 95% CI `[-12.74, -11.14]`, Wilcoxon FDR `9.88e-08`, n `25`.
- `ence` `explicit_residual_target_diffusion` minus `residual_likelihood_guided_diffusion`: mean `-11.9`, 95% CI `[-12.71, -11.14]`, Wilcoxon FDR `9.88e-08`, n `25`.
