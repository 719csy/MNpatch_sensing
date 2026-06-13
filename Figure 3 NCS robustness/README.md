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
- Posterior metric rows: `175`
- Paired statistical comparisons: `126` after masking undefined ENCE for deterministic zero-width outputs

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

| model | folds | rmse_norm_mean | rmse_kpa_mean | cov90_mean | width90_norm_mean |
|---|---:|---:|---:|---:|---:|
| unet_canonical_bxyz | 25 | 0.158806 | 9.87565 | 0.00000 | 0.00000 |
| residual_likelihood_guided_diffusion | 25 | 0.159282 | 9.90574 | 0.04028 | 0.02873 |
| prior_init_diffusion_unet | 25 | 0.159288 | 9.90593 | 0.04032 | 0.02874 |
| explicit_residual_target_diffusion | 25 | 0.161455 | 9.96194 | 0.81997 | 0.18884 |
| srcnn_bz_single_channel | 25 | 0.204006 | 11.84488 | 0.00000 | 0.00000 |
| train_mean_prior | 25 | 0.219380 | 12.24807 | n/a | n/a |
| vanilla_diffusion | 25 | 0.461796 | 14.80250 | n/a | n/a |

The strict audit uses completed high-fidelity case count, paired device benchmark completeness, five-seed posterior aggregate availability, and paired bootstrap/statistical outputs as repository-readiness gates. It is not, by itself, a scientific calibration certificate.

## Posterior Calibration Caveat

The five-seed true-OOD aggregate is internally consistent, but the interval calibration is not yet manuscript-strong for every posterior variant. `residual_likelihood_guided_diffusion` and `prior_init_diffusion_unet` have competitive RMSE but under-dispersed 90% intervals (`cov90` about `4%`, `width90_norm` about `0.029`). `explicit_residual_target_diffusion` is the best-calibrated reported probabilistic model in this package (`cov90` about `82%`, `width90_norm` about `0.189`) but still under-covers a nominal 90% interval.

Training logs include train-time diagnostic CV with `--posterior-samples 4`; the posterior aggregate reported here comes from the posterior evaluation pass with `--posterior-samples 64`. Figure panels using denser sampling should not be described as if they are the same sampling depth as train-time CV diagnostics.

For manuscript language, the current evidence supports "depth-aware reconstruction with uncertainty estimates" more strongly than fully calibrated "90% posterior recovery" for all variants. A final NCS-level claim should either use the better-calibrated explicit residual target variant for interval claims, apply/re-run calibration, or explicitly discuss the under-coverage.

## Model Accounting

The training logs contain nine model names. The posterior aggregate in this package reports seven models. The two unreported trained variants are `likelihood_guided_diffusion` and `explicit_residual_likelihood_guided_diffusion`; they are present in logs but absent from the final posterior aggregate. This package now treats that as an explicit exclusion rather than a silent omission. Before manuscript submission, those two variants should either be evaluated into the aggregate or excluded with a pre-specified rationale.

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
