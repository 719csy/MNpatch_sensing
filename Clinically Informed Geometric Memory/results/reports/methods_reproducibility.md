# Methods Reproducibility

Run from the project root:

```powershell
python .\scripts\run_cigm_experiment.py
```

Random seed: 20260606.

Generated outputs:

- `results/data_manifest.csv`
- `results/tables/geometry_descriptors_2d.csv`
- `results/tables/geometry_descriptors_3d.csv`
- `results/tables/geometry_distribution_matching.csv`
- `results/tables/independence_tests.csv`
- `results/tables/main_ablation_table.csv`
- `results/tables/ood_failure_metrics.csv`
- `results/figures/*.png` and `results/figures/*.pdf`
- `results/reports/*.md`

Expected runtime on this Windows workstation is a few minutes, dominated by reading mask CSVs and rendering 600 dpi figures.
