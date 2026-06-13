from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_METRICS = Path(
    r"H:\My Drive\MN_simulation\outputs\rspi_joint_052026\method2c_targetmap_posterior64_colab_cuda_20260531_194518\rspi_posterior_metrics.csv"
)
DEFAULT_OUT = Path(r"H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_posterior_paired_stats_20260531")


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(int(n_boot), values.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return pd.Series(out, index=p_values.index)
    order = idx[np.argsort(p[idx])]
    ranks = np.arange(1, order.size + 1)
    adjusted = p[order] * order.size / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=p_values.index)


def paired_tests(metrics: pd.DataFrame, ours_models: list[str], metric_cols: list[str], n_boot: int) -> pd.DataFrame:
    key_cols = ["dataset", "split_mode", "fold"]
    if "seed" in metrics.columns:
        key_cols.append("seed")
    available_models = sorted(metrics["model"].dropna().astype(str).unique())
    rows: list[dict] = []
    rng = np.random.default_rng(53126)
    for ours in ours_models:
        ours_df = metrics[metrics["model"].astype(str).eq(ours)][key_cols + metric_cols].copy()
        if ours_df.empty:
            continue
        for baseline in available_models:
            if baseline == ours:
                continue
            base_df = metrics[metrics["model"].astype(str).eq(baseline)][key_cols + metric_cols].copy()
            if base_df.empty:
                continue
            merged = base_df.merge(ours_df, on=key_cols, suffixes=("_baseline", "_ours"), how="inner")
            if merged.empty:
                continue
            for metric in metric_cols:
                b = pd.to_numeric(merged[f"{metric}_baseline"], errors="coerce").to_numpy(dtype=float)
                o = pd.to_numeric(merged[f"{metric}_ours"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(b) & np.isfinite(o)
                if not ok.any():
                    continue
                delta = b[ok] - o[ok]
                ci_low, ci_high = bootstrap_ci(delta, rng, n_boot=n_boot)
                if delta.size > 1:
                    try:
                        t_p = float(stats.ttest_rel(b[ok], o[ok], nan_policy="omit").pvalue)
                    except Exception:
                        t_p = float("nan")
                    try:
                        w_p = float(stats.wilcoxon(delta).pvalue) if np.any(np.abs(delta) > 0) else 1.0
                    except Exception:
                        w_p = float("nan")
                else:
                    t_p = float("nan")
                    w_p = float("nan")
                rows.append(
                    {
                        "ours_model": ours,
                        "baseline_model": baseline,
                        "metric": metric,
                        "n_pairs": int(delta.size),
                        "baseline_minus_ours_mean": float(np.mean(delta)),
                        "baseline_minus_ours_median": float(np.median(delta)),
                        "bootstrap95_low": ci_low,
                        "bootstrap95_high": ci_high,
                        "paired_t_p": t_p,
                        "wilcoxon_p": w_p,
                        "direction_good_for_ours": bool(np.mean(delta) > 0.0),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_p_fdr"] = bh_fdr(out["paired_t_p"])
        out["wilcoxon_p_fdr"] = bh_fdr(out["wilcoxon_p"])
    return out


def write_report(stats_df: pd.DataFrame, out_dir: Path, metrics_path: Path, ours_models: list[str]) -> None:
    lines = [
        "# Figure 3 Paired Posterior Statistics",
        "",
        f"- Metrics source: `{metrics_path}`.",
        f"- Ours models tested: `{', '.join(ours_models)}`.",
        "- Effect size is `baseline_minus_ours`; positive values favor ours for error metrics.",
        "",
    ]
    if stats_df.empty:
        lines.append("No paired comparisons were available.")
    else:
        top = stats_df.sort_values(["metric", "baseline_minus_ours_mean"], ascending=[True, False]).head(30)
        lines.append("## Top Comparisons")
        for row in top.itertuples(index=False):
            lines.append(
                f"- `{row.metric}` `{row.baseline_model}` minus `{row.ours_model}`: "
                f"mean `{row.baseline_minus_ours_mean:.4g}`, 95% CI "
                f"`[{row.bootstrap95_low:.4g}, {row.bootstrap95_high:.4g}]`, "
                f"Wilcoxon FDR `{row.wilcoxon_p_fdr:.3g}`, n `{row.n_pairs}`."
            )
    (out_dir / "fig3_posterior_paired_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap/statistical tests for Figure 3 posterior metrics.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ours-models", default="residual_likelihood_guided_diffusion,prior_init_diffusion_unet,explicit_residual_target_diffusion")
    parser.add_argument("--metrics-cols", default="mae_kpa,rmse_kpa,mae_norm,rmse_norm,crps_norm,uce,ence,ause_abs_error,width90_norm")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics)
    metric_cols = [c.strip() for c in args.metrics_cols.split(",") if c.strip() and c.strip() in metrics.columns]
    ours_models = [c.strip() for c in args.ours_models.split(",") if c.strip()]
    stats_df = paired_tests(metrics, ours_models, metric_cols, int(args.bootstrap))
    stats_csv = out_dir / "fig3_posterior_paired_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    summary = {
        "metrics": str(args.metrics),
        "stats_csv": str(stats_csv),
        "report_md": str(out_dir / "fig3_posterior_paired_stats.md"),
        "ours_models": ours_models,
        "metric_cols": metric_cols,
        "n_comparisons": int(len(stats_df)),
    }
    (out_dir / "fig3_posterior_paired_stats_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(stats_df, out_dir, args.metrics, ours_models)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
