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
- Paired statistical comparisons: `132`

## Key Files

- `figures/figure3_final_locked_overlay.png`
- `figures/figure3_final_locked_overlay.pdf`
- `figures/figure3_actual_data_visualization.png`
- `data/fig3/`
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

The strict audit uses the completed high-fidelity case count, paired device benchmark completeness, five-seed posterior aggregate availability, and paired bootstrap/statistical outputs as the publication-readiness gates.

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

## Excluded Large/Intermediate Files

The following were intentionally not committed:

- Full tensor cache: `method2b_512_diffusion_ready_cache.npz` (~62 MB)
- Model checkpoints under `seed_*/training/**.pt`
- Raw COMSOL exports and distributed worker intermediate folders

These are omitted to keep the repository focused on core figure evidence and analysis outputs. The manifest, audit summaries, seed posterior metrics, aggregate metrics, paired statistics, and rendered Figure 3 data package are included.
