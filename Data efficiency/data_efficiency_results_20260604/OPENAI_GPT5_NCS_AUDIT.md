```markdown
# Statistical Review of Figure 2 / Figure 3 Data-Efficiency Benchmark

## Verdict
Conditionally acceptable for internal validation summary. Not yet sufficient for publication-level quantitative claims in *Nature Computational Science*. Major gaps remain in experimental completeness and cross-device training consistency.

## Coverage
- 2D (Figure 2): includes three acquisition modalities (MMP, sheet piezo, ultrasound) and five ML baselines (SRCNN, U-Net, vanilla diffusion, DPS inverse diffusion, operator-conditioned diffusion). Coverage adequate.
- 3D (Figure 3): includes the same three devices and seven ML baselines (SRCNN proxy, scalar regression, 3D U-Net, kernel GP, vanilla/diffusion variants, operator-conditioned diffusion). Coverage formally adequate.
- Device representation: all devices appear in both 2D and 3D tables, but several are based on "fast surrogate" data rather than fully trained N-grids; this limits interchangeability.

## Metrics
- Metrics (MAE_kPa for 2D, mae_zbottom_mm for 3D) are dimensionally consistent and interpretable.
- Data-efficiency indicators (N90, AULC, DER, plateau flags) are standard for learning-curve evaluation and appear consistently defined.
- Censoring and plateau flags are useful but require validation scripts for reproducibility.
- "Common target = 10% above operator-conditioned model" is a clear reference baseline but needs justification for that threshold as statistically meaningful.

## Supported Claims
- Operator-conditioned diffusion consistently reaches target quality at lowest N across all devices; the DER ratios near unity confirm baseline anchoring.
- In several cases (e.g., MMP 2D, ultrasound 3D) other models achieve DER between 2 and 4, consistent with improved data efficiency relative to operator baseline.
- Qualitative claim supported: operator-conditioned diffusion is the most data-efficient model *within existing surrogate data*.

## Required Remaining Upgrades
1. Replace surrogate rows with full retrained N-grid benchmarks to ensure valid quantitative comparison.
2. Supply uncertainty estimates or confidence intervals for N90, DER, and AULC to enable statistical significance judgment.
3. Clarify absolute sample counts versus normalized N to verify comparability across devices.
4. Provide details on whether "operator-conditioned diffusion" uses shared weights or per-device reconditioning; current matrix cannot confirm fair cross-device usage.
5. Validate metric computation scripts against a held-out set to ensure metrics identical across 2D and 3D analyses.

Overall: design coverage complete, metrics reasonable, but data provenance and reproducibility insufficient for publication-grade data-efficiency conclusions.
```
