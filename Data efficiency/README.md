# Data Efficiency Materials

This folder contains the data-efficiency, audit, visualization, and reporting outputs for the Figure 2-3 readiness workflow in the MNpatch sensing manuscript project.

## Contents

- `data efficiency.pdf` - source planning/export document for the data-efficiency and readiness protocol.
- `pdf_pages_25_68_extracted_text.txt` - extracted text from the relevant PDF pages.
- `data/qc/` - consolidated QC tables used by the analysis.
- `fig23_data_efficiency_outputs/` - Figure 2-3 data-efficiency design files, CSV summaries, JSON counts, and runnable scripts.
- `ncs_audit_20260604/` - Nature Computational Science-oriented audit outputs, reports, API payload notes, and visualization products.
- `data_efficiency_results_20260604/` - validated data-efficiency result tables, summary reports, and generated visualizations.
- `priority1_learning_curves_060626/` - Priority 1 learning-curve-by-N outputs generated from the 2026-06-06 audit pages 68-82.

## Key Outputs

- Learning-curve and data-efficiency summaries:
  - `data/qc/fig2_metrics_by_N.csv`
  - `data/qc/fig3_metrics_by_N.csv`
  - `data/qc/learning_curve_fits.csv`
  - `data/qc/priority1_gate_update.csv`
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
- `run_priority1_learning_curves_060626.py`

## 2026-06-06 Priority 1 Learning Curves

The Priority 1 audit requirement from `dat aefficiency audit 060626.pdf` pages 68-82 is implemented by:

```text
fig23_data_efficiency_outputs/scripts/run_priority1_learning_curves_060626.py
```

It writes the required QC files:

- `data/qc/fig2_metrics_by_N.csv`
- `data/qc/fig3_metrics_by_N.csv`
- `data/qc/learning_curve_fits.csv`

It also writes:

- `data/qc/priority1_gate_update.csv`
- `data/qc/priority1_model_device_coverage_matrix.csv`
- `priority1_learning_curves_060626/PRIORITY1_LEARNING_CURVE_REPORT.md`
- `priority1_learning_curves_060626/visualizations/`

Important caveat: the requested N-grid extends beyond the currently available unique source cases/configurations, so large-N rows are marked with `sampling_mode=bootstrap_with_replacement_effective_N`.

## Notes

The local programmatic visualizations should be treated as the authoritative quantitative figures because they are generated directly from the CSV/JSON outputs. API-generated images are included for style exploration and should not replace exact data-rendered figures without verified overlays.
