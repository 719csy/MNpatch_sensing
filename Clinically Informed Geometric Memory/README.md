# Clinically Informed Geometric Memory

Reproducible experiment bundle for testing whether dermatology-image-derived geometry memory improves inverse biomechanical sensing while keeping geometry independent from stiffness, depth, thickness, background modulus, heterogeneity, noise, and operator metadata.

## Run

```powershell
python .\scripts\run_cigm_experiment.py
```

## Key Outputs

- `source_pdf_pages_32_64_extracted_text.txt`
- `results/data_manifest.csv`
- `results/tables/`
- `results/tables/topline_report_card.csv`
- `results/tables/trend_metrics.csv`
- `results/tables/actual_sample_decoupling_tests.csv`
- `results/figures/`
- `results/reports/clinical_memory_claim_check.md`
- `results/reports/piezo_failure_note.md`
- `results/reports/diffusion_vs_flow_decision.md`

The optional flow/stochastic-interpolant figure is intentionally not generated unless a true geometry-to-mechanics endpoint path is trained.
