# Data Efficiency Materials

This folder contains the data-efficiency, audit, visualization, and reporting outputs for the Figure 2-3 readiness workflow in the MNpatch sensing manuscript project.

## Contents

- `data efficiency.pdf` - source planning/export document for the data-efficiency and readiness protocol.
- `pdf_pages_25_68_extracted_text.txt` - extracted text from the relevant PDF pages.
- `data/qc/` - consolidated QC tables used by the analysis.
- `fig23_data_efficiency_outputs/` - Figure 2-3 data-efficiency design files, CSV summaries, JSON counts, and runnable scripts.
- `ncs_audit_20260604/` - Nature Computational Science-oriented audit outputs, reports, API payload notes, and visualization products.
- `data_efficiency_results_20260604/` - validated data-efficiency result tables, summary reports, and generated visualizations.

## Key Outputs

- Learning-curve and data-efficiency summaries:
  - `data/qc/fig2_metrics_by_N.csv`
  - `data/qc/fig3_metrics_by_N.csv`
  - `data/qc/fig23_N90_AULC_DER_summary.csv`
- Figure-ready visualizations:
  - `data_efficiency_results_20260604/visualizations/`
  - `ncs_audit_20260604/visualizations/`
- Audit and status reports:
  - `ncs_audit_20260604/NCS_DATA_EFFICIENCY_AUDIT_REPORT.md`
  - `data_efficiency_results_20260604/FIG23_DATA_EFFICIENCY_RESULTS_REPORT.md`
  - `data_efficiency_results_20260604/OPENAI_GPT5_NCS_AUDIT_VALIDATED.md`

## Reproducibility

The scripts used to build and validate these outputs are stored in:

```text
fig23_data_efficiency_outputs/scripts/
```

The main script entry points are:

- `build_fig23_data_efficiency_design.py`
- `build_fig23_data_efficiency_results.py`
- `run_ncs_data_efficiency_audit.py`
- `generate_api_visualizations.py`
- `generate_fig23_api_visualizations.py`

## Notes

The local programmatic visualizations should be treated as the authoritative quantitative figures because they are generated directly from the CSV/JSON outputs. API-generated images are included for style exploration and should not replace exact data-rendered figures without verified overlays.
