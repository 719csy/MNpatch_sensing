# Figure 3 NCS Robustness Package

This directory contains the core data, analysis outputs, scripts, and manuscript-ready figure artifacts for Figure 3:

**3D domain-randomized skin mechanics recover depth-aware latent biomechanical posteriors**

## Status

- Strict audit gate: `PASS`
- Audit reason: `minimum high-fidelity, paired benchmark and posterior aggregate gates passed`
- High-fidelity Method2B cases: `512 / 512`
- Strain rows: `1536 / 1536`
- Paired MMP / piezo / ultrasound benchmark rows: `150 / 150`
- Full512 posterior training seeds: `5`
- Posterior metric rows: `375`
- Posterior aggregate models: `15`
- Posterior evaluation samples: `32`
- Paired statistical comparisons: `1190` after masking undefined ENCE for deterministic zero-width outputs

## Key Files

- `figures/figure3_final_locked_overlay.png`
- `figures/figure3_final_locked_overlay.pdf`
- `figures/figure3_actual_data_visualization.png`
- `data/fig3/`
- `data/shared_posterior/fig3_to_fig4_posterior_bank.npz`
- `data/shared_posterior/fig3_to_fig4_manifest.json`
- `outputs/audit/fig3_ncs_robustness_audit_summary.json`
- `outputs/posterior_aggregate/method2b_512_five_seed_posterior_metrics.csv`
- `outputs/posterior_aggregate/method2b_512_five_seed_posterior_aggregate.csv`
- `outputs/paired_stats/fig3_posterior_paired_stats.csv`
- `outputs/seed_metrics/seed_*/rspi_posterior_metrics.csv`
- `file_manifest.json`

## Aggregate Snapshot

Five-seed true-OOD posterior aggregate, sorted by `rmse_norm_mean`:

| model | folds | rmse_norm_mean | rmse_kpa_mean | cov90_mean | width90_norm_mean | ence_mean |
|---|---:|---:|---:|---:|---:|---:|
| unet_canonical_bxyz | 25 | 0.15881 | 9.87565 | 0.00000 | 0.00000 | n/a |
| prior_init_diffusion_unet | 25 | 0.15928 | 9.90609 | 0.03935 | 0.02762 | 13.78102 |
| prior_init_diffusion_unet_calibrated90 | 25 | 0.15928 | 9.90609 | 0.90646 | 0.35058 | 0.25983 |
| residual_likelihood_guided_diffusion_calibrated90 | 25 | 0.15929 | 9.90550 | 0.90648 | 0.35060 | 0.25730 |
| residual_likelihood_guided_diffusion | 25 | 0.15929 | 9.90550 | 0.03923 | 0.02762 | 13.77027 |
| explicit_residual_likelihood_guided_diffusion | 25 | 0.16178 | 9.96949 | 0.80279 | 0.17993 | 1.98816 |
| explicit_residual_likelihood_guided_diffusion_calibrated90 | 25 | 0.16178 | 9.96949 | 0.90686 | 0.31879 | 0.71341 |
| explicit_residual_target_diffusion | 25 | 0.16189 | 9.96433 | 0.80268 | 0.18040 | 1.99252 |
| explicit_residual_target_diffusion_calibrated90 | 25 | 0.16189 | 9.96433 | 0.90737 | 0.32123 | 0.72463 |
| srcnn_bz_single_channel | 25 | 0.20401 | 11.84488 | 0.00000 | 0.00000 | n/a |
| train_mean_prior | 25 | 0.21938 | 12.24807 | n/a | n/a | n/a |
| vanilla_diffusion_calibrated90 | 25 | 0.46164 | 14.80282 | 0.89824 | 2.23184 | 0.39280 |
| vanilla_diffusion | 25 | 0.46164 | 14.80282 | 0.04963 | 0.25065 | 5.17810 |
| likelihood_guided_diffusion | 25 | 0.46180 | 14.80810 | 0.04918 | 0.25027 | 5.18632 |
| likelihood_guided_diffusion_calibrated90 | 25 | 0.46180 | 14.80810 | 0.89583 | 2.24031 | 0.39338 |

The strict audit uses completed high-fidelity case count, paired device benchmark completeness, five-seed posterior aggregate availability, and paired bootstrap/statistical outputs as repository-readiness gates. It is not, by itself, a scientific calibration certificate.

## Posterior Calibration Caveat

The five-seed true-OOD aggregate is internally consistent and now includes validation-calibrated 90% interval variants. The uncalibrated residual/prior-initialized variants remain sharply under-dispersed (`cov90` about `4%`, `width90_norm` about `0.028`), so they should be described as low-RMSE reconstructions rather than calibrated posteriors. The `_calibrated90` variants reach the intended 90% coverage on the held-out true-OOD aggregate (`cov90` about `0.906` to `0.907` for prior/residual/explicit residual variants) with wider intervals.

Training logs include train-time diagnostic CV with `--posterior-samples 4`; the manuscript aggregate reported here comes from the recalibrated posterior evaluation pass with `--posterior-samples 32`. Figure panels using denser visualization samples should not be described as if they are the same sampling depth as train-time CV diagnostics.

For manuscript language, use the calibrated variants for posterior coverage claims and the uncalibrated variants only for point-reconstruction/error comparisons. Vanilla diffusion can also be calibrated to about 90% coverage, but only by expanding intervals roughly sevenfold relative to the calibrated residual/prior variants.

## Model Accounting

The posterior aggregate now includes all nine trained/evaluated model families plus validation-calibrated 90% variants where posterior sampling supports interval calibration. The previously omitted `likelihood_guided_diffusion` and `explicit_residual_likelihood_guided_diffusion` variants are included, along with their `_calibrated90` counterparts. Paired bootstrap/Wilcoxon statistics test the full non-vanilla diffusion posterior family rather than a hand-picked subset.

## Metric Handling

`ENCE` is undefined for deterministic zero-width outputs. Earlier paired statistics included enormous ENCE values for deterministic baselines because the normalization denominator was effectively zero. The committed paired statistics now mask ENCE where `width90_norm <= 1e-12` or `posterior_std_norm_mean <= 1e-12` before testing; error metrics, coverage, and width metrics remain unchanged.

## Reproduction Notes

The local finalization entry point is:

```powershell
py -3.12 "scripts\fig3_full512_finalize.py"
```

The main aggregation and audit scripts copied here are:

- `scripts/method2b_512_collect_posterior_aggregate.py`
- `scripts/fig3_posterior_paired_stats.py`
- `scripts/fig3_ncs_robustness_audit.py`
- `scripts/fig3_render_realdata_dashboard.py`
- `scripts/fig3_build_hybrid_realdata_on_v3.py`
- `scripts/fig3_full512_finalize.py`

The Colab orchestration scripts used for A100 five-seed training are in `colab/`.

Some audit and provenance files retain author-workstation absolute paths under `H:\My Drive\...` to identify the original COMSOL/Colab run locations. Reproducible package contents live under this directory; see `file_manifest.json` and `outputs/audit/path_map.json` for the package mapping.

## Excluded Large/Intermediate Files

The following were intentionally not committed:

- Full tensor cache: `method2b_512_diffusion_ready_cache.npz` (~62 MB)
- Model checkpoints under `seed_*/training/**.pt`
- Raw COMSOL exports and distributed worker intermediate folders

These are omitted to keep the repository focused on core figure evidence and analysis outputs. The manifest, audit summaries, seed posterior metrics, aggregate metrics, paired statistics, and rendered Figure 3 data package are included.
