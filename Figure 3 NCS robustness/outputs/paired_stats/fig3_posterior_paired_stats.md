# Figure 3 Paired Posterior Statistics

- Metrics source: `H:\My Drive\MN_simulation\outputs\rspi_joint_052026\method2b_512_five_seed_ood_full512_20260611\aggregate\method2b_512_five_seed_posterior_metrics.csv`.
- Ours models tested: `likelihood_guided_diffusion, likelihood_guided_diffusion_calibrated90, prior_init_diffusion_unet, prior_init_diffusion_unet_calibrated90, residual_likelihood_guided_diffusion, residual_likelihood_guided_diffusion_calibrated90, explicit_residual_target_diffusion, explicit_residual_target_diffusion_calibrated90, explicit_residual_likelihood_guided_diffusion, explicit_residual_likelihood_guided_diffusion_calibrated90`.
- Effect size is `baseline_minus_ours`; positive values favor ours for error metrics.

## Top Comparisons
- `ause_abs_error` `vanilla_diffusion` minus `residual_likelihood_guided_diffusion_calibrated90`: mean `0.07505`, 95% CI `[0.07035, 0.07946]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion_calibrated90` minus `residual_likelihood_guided_diffusion_calibrated90`: mean `0.07505`, 95% CI `[0.07044, 0.0795]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion` minus `residual_likelihood_guided_diffusion`: mean `0.07505`, 95% CI `[0.07048, 0.07946]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion_calibrated90` minus `residual_likelihood_guided_diffusion`: mean `0.07505`, 95% CI `[0.07052, 0.07958]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion` minus `prior_init_diffusion_unet`: mean `0.075`, 95% CI `[0.07044, 0.07951]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion` minus `prior_init_diffusion_unet_calibrated90`: mean `0.075`, 95% CI `[0.07044, 0.07941]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion_calibrated90` minus `prior_init_diffusion_unet`: mean `0.075`, 95% CI `[0.07036, 0.07947]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion_calibrated90` minus `prior_init_diffusion_unet_calibrated90`: mean `0.075`, 95% CI `[0.07035, 0.0794]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion_calibrated90` minus `residual_likelihood_guided_diffusion_calibrated90`: mean `0.07473`, 95% CI `[0.07021, 0.07896]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion` minus `residual_likelihood_guided_diffusion_calibrated90`: mean `0.07473`, 95% CI `[0.07026, 0.07893]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion_calibrated90` minus `residual_likelihood_guided_diffusion`: mean `0.07473`, 95% CI `[0.07024, 0.07892]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion` minus `residual_likelihood_guided_diffusion`: mean `0.07473`, 95% CI `[0.07028, 0.07899]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion_calibrated90` minus `prior_init_diffusion_unet`: mean `0.07468`, 95% CI `[0.0702, 0.07892]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion` minus `prior_init_diffusion_unet`: mean `0.07468`, 95% CI `[0.07034, 0.07895]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion_calibrated90` minus `prior_init_diffusion_unet_calibrated90`: mean `0.07468`, 95% CI `[0.0702, 0.07891]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion` minus `prior_init_diffusion_unet_calibrated90`: mean `0.07468`, 95% CI `[0.07018, 0.07889]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `srcnn_bz_single_channel` minus `residual_likelihood_guided_diffusion_calibrated90`: mean `0.07083`, 95% CI `[0.06839, 0.07352]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `srcnn_bz_single_channel` minus `residual_likelihood_guided_diffusion`: mean `0.07083`, 95% CI `[0.06848, 0.07351]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `srcnn_bz_single_channel` minus `prior_init_diffusion_unet`: mean `0.07078`, 95% CI `[0.06827, 0.07341]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `srcnn_bz_single_channel` minus `prior_init_diffusion_unet_calibrated90`: mean `0.07078`, 95% CI `[0.06833, 0.07345]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion` minus `explicit_residual_target_diffusion`: mean `0.0699`, 95% CI `[0.06546, 0.07439]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion_calibrated90` minus `explicit_residual_target_diffusion`: mean `0.0699`, 95% CI `[0.06554, 0.07444]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion` minus `explicit_residual_target_diffusion_calibrated90`: mean `0.0699`, 95% CI `[0.06556, 0.07429]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion_calibrated90` minus `explicit_residual_target_diffusion_calibrated90`: mean `0.0699`, 95% CI `[0.06549, 0.07434]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion_calibrated90` minus `explicit_residual_target_diffusion`: mean `0.06959`, 95% CI `[0.06536, 0.07388]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion` minus `explicit_residual_target_diffusion`: mean `0.06959`, 95% CI `[0.06532, 0.07386]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion_calibrated90` minus `explicit_residual_target_diffusion_calibrated90`: mean `0.06959`, 95% CI `[0.06528, 0.07382]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `likelihood_guided_diffusion` minus `explicit_residual_target_diffusion_calibrated90`: mean `0.06959`, 95% CI `[0.06532, 0.07378]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion` minus `explicit_residual_likelihood_guided_diffusion`: mean `0.06953`, 95% CI `[0.06513, 0.07382]`, Wilcoxon FDR `9.77e-08`, n `25`.
- `ause_abs_error` `vanilla_diffusion` minus `explicit_residual_likelihood_guided_diffusion_calibrated90`: mean `0.06953`, 95% CI `[0.06519, 0.07389]`, Wilcoxon FDR `9.77e-08`, n `25`.
