# Figure 3 Paired Posterior Statistics

- Metrics source: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\method2b_512_five_seed_ood_full512_20260611\aggregate\method2b_512_five_seed_posterior_metrics.csv`.
- Ours models tested: `residual_likelihood_guided_diffusion, prior_init_diffusion_unet, explicit_residual_target_diffusion`.
- Effect size is `baseline_minus_ours`; positive values favor ours for error metrics.

## Top Comparisons
- `ause_abs_error` `srcnn_bz_single_channel` minus `residual_likelihood_guided_diffusion`: mean `0.07155`, 95% CI `[0.06921, 0.07415]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ause_abs_error` `srcnn_bz_single_channel` minus `prior_init_diffusion_unet`: mean `0.07155`, 95% CI `[0.0692, 0.07415]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ause_abs_error` `srcnn_bz_single_channel` minus `explicit_residual_target_diffusion`: mean `0.06683`, 95% CI `[0.06462, 0.0693]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ause_abs_error` `unet_canonical_bxyz` minus `residual_likelihood_guided_diffusion`: mean `0.04616`, 95% CI `[0.04448, 0.04791]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ause_abs_error` `unet_canonical_bxyz` minus `prior_init_diffusion_unet`: mean `0.04615`, 95% CI `[0.0445, 0.04788]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ause_abs_error` `unet_canonical_bxyz` minus `explicit_residual_target_diffusion`: mean `0.04144`, 95% CI `[0.03945, 0.04333]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ause_abs_error` `explicit_residual_target_diffusion` minus `residual_likelihood_guided_diffusion`: mean `0.004729`, 95% CI `[0.003751, 0.005767]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ause_abs_error` `explicit_residual_target_diffusion` minus `prior_init_diffusion_unet`: mean `0.00472`, 95% CI `[0.003758, 0.005731]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ause_abs_error` `prior_init_diffusion_unet` minus `residual_likelihood_guided_diffusion`: mean `9.594e-06`, 95% CI `[-4.948e-05, 6.974e-05]`, Wilcoxon FDR `0.697`, n `25`.
- `ause_abs_error` `residual_likelihood_guided_diffusion` minus `prior_init_diffusion_unet`: mean `-9.594e-06`, 95% CI `[-6.901e-05, 4.974e-05]`, Wilcoxon FDR `0.697`, n `25`.
- `ause_abs_error` `prior_init_diffusion_unet` minus `explicit_residual_target_diffusion`: mean `-0.00472`, 95% CI `[-0.005734, -0.003768]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ause_abs_error` `residual_likelihood_guided_diffusion` minus `explicit_residual_target_diffusion`: mean `-0.004729`, 95% CI `[-0.005784, -0.003761]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `crps_norm` `srcnn_bz_single_channel` minus `explicit_residual_target_diffusion`: mean `0.057`, 95% CI `[0.05298, 0.06055]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `crps_norm` `srcnn_bz_single_channel` minus `residual_likelihood_guided_diffusion`: mean `0.04363`, 95% CI `[0.04104, 0.04608]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `crps_norm` `srcnn_bz_single_channel` minus `prior_init_diffusion_unet`: mean `0.04363`, 95% CI `[0.041, 0.04608]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `crps_norm` `unet_canonical_bxyz` minus `explicit_residual_target_diffusion`: mean `0.01606`, 95% CI `[0.0128, 0.01859]`, Wilcoxon FDR `6.33e-07`, n `25`.
- `crps_norm` `prior_init_diffusion_unet` minus `explicit_residual_target_diffusion`: mean `0.01337`, 95% CI `[0.01015, 0.01585]`, Wilcoxon FDR `3.52e-06`, n `25`.
- `crps_norm` `residual_likelihood_guided_diffusion` minus `explicit_residual_target_diffusion`: mean `0.01337`, 95% CI `[0.01018, 0.01586]`, Wilcoxon FDR `3.52e-06`, n `25`.
- `crps_norm` `unet_canonical_bxyz` minus `residual_likelihood_guided_diffusion`: mean `0.002688`, 95% CI `[0.002429, 0.00295]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `crps_norm` `unet_canonical_bxyz` minus `prior_init_diffusion_unet`: mean `0.002687`, 95% CI `[0.002427, 0.002951]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `crps_norm` `prior_init_diffusion_unet` minus `residual_likelihood_guided_diffusion`: mean `9.495e-07`, 95% CI `[-9.88e-06, 1.126e-05]`, Wilcoxon FDR `0.568`, n `25`.
- `crps_norm` `residual_likelihood_guided_diffusion` minus `prior_init_diffusion_unet`: mean `-9.495e-07`, 95% CI `[-1.154e-05, 1e-05]`, Wilcoxon FDR `0.568`, n `25`.
- `crps_norm` `explicit_residual_target_diffusion` minus `residual_likelihood_guided_diffusion`: mean `-0.01337`, 95% CI `[-0.01585, -0.01011]`, Wilcoxon FDR `3.52e-06`, n `25`.
- `crps_norm` `explicit_residual_target_diffusion` minus `prior_init_diffusion_unet`: mean `-0.01337`, 95% CI `[-0.01583, -0.01014]`, Wilcoxon FDR `3.52e-06`, n `25`.
- `ence` `srcnn_bz_single_channel` minus `explicit_residual_target_diffusion`: mean `2.019e+07`, 95% CI `[1.987e+07, 2.048e+07]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ence` `srcnn_bz_single_channel` minus `prior_init_diffusion_unet`: mean `2.019e+07`, 95% CI `[1.986e+07, 2.048e+07]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ence` `srcnn_bz_single_channel` minus `residual_likelihood_guided_diffusion`: mean `2.019e+07`, 95% CI `[1.987e+07, 2.048e+07]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ence` `unet_canonical_bxyz` minus `explicit_residual_target_diffusion`: mean `1.567e+07`, 95% CI `[1.529e+07, 1.608e+07]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ence` `unet_canonical_bxyz` minus `prior_init_diffusion_unet`: mean `1.567e+07`, 95% CI `[1.529e+07, 1.609e+07]`, Wilcoxon FDR `9.59e-08`, n `25`.
- `ence` `unet_canonical_bxyz` minus `residual_likelihood_guided_diffusion`: mean `1.567e+07`, 95% CI `[1.53e+07, 1.608e+07]`, Wilcoxon FDR `9.59e-08`, n `25`.
