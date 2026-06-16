from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


EPS = 1e-8
SCENARIO_LABELS = {
    "iid_test": "ID held-out",
    "ood_unseen_shape": "Morphology OOD",
    "ood_unseen_depth_range": "Depth OOD",
    "ood_unseen_layer_modulus": "Mechanical-background OOD",
    "ood_unseen_operator_noise": "Noise/drift OOD",
}
SELECTED_CASES = [
    ("fig4b_irregular_morphology", "ood_unseen_shape", "Irregular morphology"),
    ("fig4b_deep_lesion", "ood_unseen_depth_range", "Deep lesion"),
    ("fig4b_noisy_input", "ood_unseen_operator_noise", "Noisy input"),
    ("fig4b_missing_channels", "iid_test", "Missing channels"),
]


def as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else str(x))
    columns = [str(c) for c in text.columns]
    rows = text.values.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "metrics": out_dir / "metrics",
        "panels": out_dir / "panels",
        "arrays": out_dir / "arrays",
        "selected_cases": out_dir / "selected_cases",
        "provenance": out_dir / "provenance",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def scenario_label(key: str) -> str:
    return SCENARIO_LABELS.get(str(key), str(key).replace("_", " ").title())


def method_label(model: str, cfg: dict[str, Any]) -> str:
    return cfg["models"]["figure_methods"].get(model, model)


def sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray([np.nan if v is None else float(v) for v in p_values], dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return out.tolist()
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / float(i + 1))
        adjusted[i] = running
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out.tolist()


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if len(np.unique(labels)) < 2:
        return float("nan")
    if len(np.unique(scores)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def load_source_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    paths = cfg["paths"]
    cache_dir = as_path(paths["cache_dir"])
    manifest_path = cache_dir / cfg["data"]["manifest_csv"]
    cache_path = cache_dir / cfg["data"]["cache_npz"]
    summary_path = cache_dir / "method2b_512_diffusion_cache_summary.json"

    manifest = pd.read_csv(manifest_path).reset_index(drop=True)
    with np.load(cache_path, allow_pickle=True) as data:
        cache = {key: np.asarray(data[key]) for key in data.files}
    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig") as handle:
            summary = json.load(handle)
    return manifest, cache, summary


def audit_real_anchors(cfg: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    cached = dirs["metrics"] / "fig4_real_anchor_missing_manifest.csv"
    if cached.exists():
        return pd.read_csv(cached)

    roots = [as_path(cfg["paths"]["mn_root"]), as_path(cfg["paths"]["clinical_ontology_root"])]
    anchor_specs = {
        "phantom": ["phantom"],
        "animal": ["animal", "rat", "mouse", "murine"],
        "ex_vivo": ["ex_vivo", "ex vivo", "excised"],
        "clinical_complex": ["clinical_complex", "clinical complex"],
    }
    rows = []
    for anchor_type, tokens in anchor_specs.items():
        candidates: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                name = str(path).lower()
                if not any(token in name for token in tokens):
                    continue
                if path.suffix.lower() in {".npy", ".npz", ".mat", ".csv", ".json"}:
                    candidates.append(path)
                if len(candidates) >= 20:
                    break
            if len(candidates) >= 20:
                break
        rows.append(
            {
                "anchor_type": anchor_type,
                "found_candidate_file_count": len(candidates),
                "status": "candidate_files_found_needs_manual_schema_check" if candidates else "missing",
                "candidate_examples": ";".join(str(p) for p in candidates[:5]),
                "required_note": "Only B8/reference-map schema-compatible files should be treated as real anchors.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["metrics"] / "fig4_real_anchor_missing_manifest.csv", index=False)
    return out


def load_seed_metrics(cfg: dict[str, Any], dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root = as_path(cfg["paths"]["run_root"])
    rows = []
    status_rows = []
    for seed in cfg["data"]["expected_seeds"]:
        seed_dir = run_root / f"seed_{seed}"
        posterior_csv = seed_dir / "posterior" / "rspi_posterior_metrics.csv"
        posterior_aggregate = seed_dir / "posterior" / "rspi_posterior_aggregate.csv"
        training_csv = seed_dir / "training" / "step3j_canonical_cv_metrics.csv"
        complete = posterior_csv.exists() and training_csv.exists()
        status_rows.append(
            {
                "seed": seed,
                "seed_dir": str(seed_dir),
                "posterior_metrics_found": posterior_csv.exists(),
                "posterior_aggregate_found": posterior_aggregate.exists(),
                "training_metrics_found": training_csv.exists(),
                "status": "complete" if complete else "missing_or_incomplete",
            }
        )
        if not posterior_csv.exists():
            continue
        df = pd.read_csv(posterior_csv)
        df["seed"] = int(seed)
        df["source_metrics_path"] = str(posterior_csv)
        rows.append(df)
    status = pd.DataFrame(status_rows)
    status.to_csv(dirs["provenance"] / "fig4_seed_status.csv", index=False)
    if not rows:
        raise RuntimeError(f"No posterior metrics were found under {run_root}")
    metrics = pd.concat(rows, ignore_index=True)
    metrics["scenario_key"] = metrics["test_group"].astype(str)
    metrics["scenario_label"] = metrics["scenario_key"].map(scenario_label)
    metrics["method_label"] = metrics["model"].map(lambda model: method_label(model, cfg))
    metrics.to_csv(dirs["metrics"] / "fig4_fold_level_metrics.csv", index=False)
    return metrics, status


def write_manifest_outputs(
    cfg: dict[str, Any],
    manifest: pd.DataFrame,
    cache: dict[str, np.ndarray],
    cache_summary: dict[str, Any],
    seed_status: pd.DataFrame,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    manifest_out = manifest.copy()
    manifest_out["scenario_key"] = np.where(
        manifest_out["split_role"].astype(str).eq("iid_test"),
        "iid_test",
        manifest_out["ood_split_family"].astype(str),
    )
    manifest_out["scenario_label"] = manifest_out["scenario_key"].map(scenario_label)
    manifest_out.to_csv(dirs["metrics"] / "fig4_ood_manifest.csv", index=False)

    split_summary = (
        manifest_out.groupby(["split_role", "scenario_key", "scenario_label"], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .sort_values(["split_role", "scenario_key"])
    )
    split_summary.to_csv(dirs["metrics"] / "id_ood_split_summary.csv", index=False)
    write_json(
        dirs["metrics"] / "id_ood_split_summary.json",
        {
            "sample_count": int(len(manifest_out)),
            "input_shape": list(cache["input64"].shape) if "input64" in cache else None,
            "target_shape": list(cache["target_kpa64"].shape) if "target_kpa64" in cache else None,
            "split_role_counts": manifest_out["split_role"].value_counts().to_dict(),
            "ood_family_counts": manifest_out["ood_split_family"].value_counts().to_dict(),
            "cache_summary": cache_summary,
            "seed_status": seed_status.to_dict(orient="records"),
        },
    )

    taxonomy_rows = []
    for key, item in cfg["ood_taxonomy"].items():
        family = item.get("current_family") or ""
        n_manifest = int((manifest_out["ood_split_family"].astype(str) == family).sum()) if family else 0
        taxonomy_rows.append(
            {
                "taxonomy_key": key,
                "figure_label": item["figure_label"],
                "current_family": family,
                "status": item["status"],
                "n_manifest_samples": n_manifest,
                "evidence": "current full512 manifest" if n_manifest else "not present in current full512 manifest",
            }
        )
    taxonomy = pd.DataFrame(taxonomy_rows)
    taxonomy.to_csv(dirs["metrics"] / "fig4_ood_taxonomy.csv", index=False)
    return taxonomy


def summarize_metrics(cfg: dict[str, Any], metrics: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    figure_models = set(cfg["models"]["figure_methods"].keys())
    fig_metrics = metrics[metrics["model"].isin(figure_models)].copy()

    numeric_cols = [
        "mae_norm",
        "rmse_norm",
        "psnr_norm",
        "ssim_norm",
        "mae_kpa",
        "rmse_kpa",
        "spearman_pixel",
        "crps_norm",
        "posterior_std_norm_mean",
        "uncertainty_error_spearman",
        "failure_auroc_top10_abs_error",
        "uce",
        "ause_abs_error",
        "pit_mean",
    ]
    rows = []
    for keys, sub in fig_metrics.groupby(["model", "method_label", "scenario_key", "scenario_label"], sort=False):
        model, label, scenario_key, scen_label = keys
        row: dict[str, Any] = {
            "model": model,
            "method_label": label,
            "scenario_key": scenario_key,
            "scenario_label": scen_label,
            "n_rows": int(len(sub)),
            "n_seeds": int(sub["seed"].nunique()),
        }
        for col in numeric_cols:
            if col in sub:
                vals = pd.to_numeric(sub[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
                row[f"{col}_std"] = float(vals.std(ddof=0)) if vals.notna().sum() > 1 else 0.0
                row[f"{col}_sem"] = sem(vals)
        rows.append(row)
    perf = pd.DataFrame(rows)
    perf.to_csv(dirs["metrics"] / "fig4_performance_by_method_scenario.csv", index=False)

    retention_detail = []
    id_rows = fig_metrics[fig_metrics["scenario_key"].eq("iid_test")].set_index(["seed", "model"])
    for _, row in fig_metrics[~fig_metrics["scenario_key"].eq("iid_test")].iterrows():
        key = (row["seed"], row["model"])
        if key not in id_rows.index:
            continue
        id_row = id_rows.loc[key]
        item = {
            "seed": int(row["seed"]),
            "model": row["model"],
            "method_label": row["method_label"],
            "scenario_key": row["scenario_key"],
            "scenario_label": row["scenario_label"],
        }
        for metric in cfg["metrics"]["higher_better"]:
            denom = float(id_row.get(metric, np.nan))
            numer = float(row.get(metric, np.nan))
            item[f"{metric}_retention"] = numer / denom if np.isfinite(numer) and abs(denom) > EPS else np.nan
        for metric in cfg["metrics"]["lower_better"]:
            denom = float(row.get(metric, np.nan))
            numer = float(id_row.get(metric, np.nan))
            item[f"{metric}_retention"] = numer / denom if np.isfinite(numer) and abs(denom) > EPS else np.nan
        retention_detail.append(item)
    retention = pd.DataFrame(retention_detail)
    retention.to_csv(dirs["metrics"] / "fig4_performance_retention_detail.csv", index=False)

    ret_rows = []
    ret_metric_cols = [c for c in retention.columns if c.endswith("_retention")]
    for keys, sub in retention.groupby(["model", "method_label", "scenario_key", "scenario_label"], sort=False):
        model, label, scenario_key, scen_label = keys
        out = {
            "model": model,
            "method_label": label,
            "scenario_key": scenario_key,
            "scenario_label": scen_label,
            "n_seeds": int(sub["seed"].nunique()),
        }
        for col in ret_metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            out[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            out[f"{col}_std"] = float(vals.std(ddof=0)) if vals.notna().sum() > 1 else 0.0
        ret_rows.append(out)
    retention_summary = pd.DataFrame(ret_rows)
    retention_summary.to_csv(dirs["metrics"] / "fig4_performance_retention.csv", index=False)

    uncertainty_shift = compute_uncertainty_shift(cfg, fig_metrics)
    uncertainty_shift.to_csv(dirs["metrics"] / "fig4_uncertainty_shift.csv", index=False)
    uncertainty_shift[["model", "method_label", "scenario_key", "scenario_label", "id_vs_ood_auroc", "granularity"]].to_csv(
        dirs["metrics"] / "fig4_ood_detection_auroc.csv", index=False
    )

    failure_rows = []
    for keys, sub in fig_metrics.groupby(["model", "method_label", "scenario_key", "scenario_label"], sort=False):
        model, label, scenario_key, scen_label = keys
        vals = pd.to_numeric(sub["failure_auroc_top10_abs_error"], errors="coerce")
        failure_rows.append(
            {
                "model": model,
                "method_label": label,
                "scenario_key": scenario_key,
                "scenario_label": scen_label,
                "failure_auroc_mean": float(vals.mean()) if vals.notna().any() else np.nan,
                "failure_auroc_std": float(vals.std(ddof=0)) if vals.notna().sum() > 1 else 0.0,
                "n_rows": int(vals.notna().sum()),
                "definition": cfg["metrics"]["failure_definition"]["note"],
            }
        )
    failure = pd.DataFrame(failure_rows)
    failure.to_csv(dirs["metrics"] / "fig4_failure_detection_auroc.csv", index=False)

    risk_coverage = compute_risk_coverage(cfg, fig_metrics)
    risk_coverage.to_csv(dirs["metrics"] / "fig4_risk_coverage.csv", index=False)

    return {
        "figure_metrics": fig_metrics,
        "performance": perf,
        "retention": retention_summary,
        "uncertainty_shift": uncertainty_shift,
        "failure": failure,
        "risk_coverage": risk_coverage,
    }


def compute_uncertainty_shift(cfg: dict[str, Any], metrics: pd.DataFrame) -> pd.DataFrame:
    score_col = cfg["metrics"]["uncertainty_score"]
    rows = []
    p_values = []
    for (model, label), sub in metrics.groupby(["model", "method_label"], sort=False):
        id_scores = pd.to_numeric(sub.loc[sub["scenario_key"].eq("iid_test"), score_col], errors="coerce").dropna()
        for (scenario_key, scenario_label_value), ood_sub in sub[~sub["scenario_key"].eq("iid_test")].groupby(
            ["scenario_key", "scenario_label"], sort=False
        ):
            ood_scores = pd.to_numeric(ood_sub[score_col], errors="coerce").dropna()
            p = np.nan
            if len(id_scores) > 0 and len(ood_scores) > 0:
                try:
                    p = float(mannwhitneyu(id_scores, ood_scores, alternative="two-sided").pvalue)
                except ValueError:
                    p = np.nan
            labels = np.r_[np.zeros(len(id_scores), dtype=int), np.ones(len(ood_scores), dtype=int)]
            scores = np.r_[id_scores.to_numpy(dtype=float), ood_scores.to_numpy(dtype=float)]
            rows.append(
                {
                    "model": model,
                    "method_label": label,
                    "scenario_key": scenario_key,
                    "scenario_label": scenario_label_value,
                    "id_median_uncertainty": float(id_scores.median()) if len(id_scores) else np.nan,
                    "ood_median_uncertainty": float(ood_scores.median()) if len(ood_scores) else np.nan,
                    "median_shift_ood_minus_id": (
                        float(ood_scores.median() - id_scores.median()) if len(id_scores) and len(ood_scores) else np.nan
                    ),
                    "mannwhitney_p": p,
                    "id_vs_ood_auroc": safe_auc(labels, scores) if len(labels) else np.nan,
                    "n_id": int(len(id_scores)),
                    "n_ood": int(len(ood_scores)),
                    "granularity": "seed_test_group",
                }
            )
            p_values.append(p)
    adjusted = bh_adjust(p_values)
    for row, p_adj in zip(rows, adjusted):
        row["bh_q"] = p_adj
    return pd.DataFrame(rows)


def compute_risk_coverage(cfg: dict[str, Any], metrics: pd.DataFrame) -> pd.DataFrame:
    score_col = cfg["metrics"]["uncertainty_score"]
    rows = []
    for (model, label), sub in metrics.groupby(["model", "method_label"], sort=False):
        risk = 1.0 - pd.to_numeric(sub["ssim_norm"], errors="coerce")
        score = pd.to_numeric(sub[score_col], errors="coerce")
        valid = risk.notna()
        if valid.sum() == 0:
            continue
        risk_vals = risk[valid].to_numpy(dtype=float)
        score_vals = score[valid].to_numpy(dtype=float)
        finite_score = np.isfinite(score_vals)
        uncertainty_available = bool(finite_score.any())
        if not uncertainty_available:
            order = np.arange(len(risk_vals))
        else:
            fill = np.nanmax(score_vals[finite_score]) + 1.0 if finite_score.any() else 0.0
            order = np.argsort(np.where(finite_score, score_vals, fill))
        n = len(risk_vals)
        for coverage in cfg["metrics"]["coverage_grid"]:
            keep = max(1, int(math.ceil(float(coverage) * n)))
            chosen = order[:keep]
            rows.append(
                {
                    "model": model,
                    "method_label": label,
                    "coverage": float(coverage),
                    "risk_1_minus_ssim": float(np.mean(risk_vals[chosen])),
                    "n_accepted_groups": int(keep),
                    "n_total_groups": int(n),
                    "uncertainty_available": uncertainty_available,
                    "granularity": "seed_test_group",
                }
            )
    return pd.DataFrame(rows)


def plot_taxonomy(taxonomy: pd.DataFrame, dirs: dict[str, Path], cfg: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    colors = taxonomy["status"].map(
        {
            "available_existing": "#2a9d8f",
            "representative_case_synthetic_perturbation_only": "#e9c46a",
            "not_available_in_current_full512_metrics": "#b0b7c3",
        }
    ).fillna("#b0b7c3")
    y = np.arange(len(taxonomy))
    ax.barh(y, taxonomy["n_manifest_samples"], color=colors, edgecolor="#1f2937", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(taxonomy["figure_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Samples in current full512 OOD cache", fontsize=8)
    ax.set_title("Figure 4a. OOD taxonomy and current data availability", fontsize=10, weight="bold")
    for i, row in taxonomy.iterrows():
        ax.text(max(float(row["n_manifest_samples"]), 1.0) + 2.0, i, row["status"].replace("_", " "), va="center", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(dirs["panels"] / "fig4a_ood_taxonomy.png", dpi=int(cfg["plotting"]["dpi"]))
    fig.savefig(dirs["panels"] / "fig4a_ood_taxonomy.svg")
    plt.close(fig)


def plot_retention(retention: pd.DataFrame, dirs: dict[str, Path], cfg: dict[str, Any]) -> None:
    if retention.empty or "ssim_norm_retention_mean" not in retention:
        return
    order_models = list(cfg["models"]["figure_methods"].keys())
    order_scenarios = [
        "ood_unseen_shape",
        "ood_unseen_depth_range",
        "ood_unseen_layer_modulus",
        "ood_unseen_operator_noise",
    ]
    pivot = retention.pivot_table(
        index="method_label",
        columns="scenario_label",
        values="ssim_norm_retention_mean",
        aggfunc="mean",
    )
    method_order = [cfg["models"]["figure_methods"][m] for m in order_models if cfg["models"]["figure_methods"][m] in pivot.index]
    scen_order = [scenario_label(s) for s in order_scenarios if scenario_label(s) in pivot.columns]
    pivot = pivot.reindex(index=method_order, columns=scen_order)

    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=0.0, vmax=max(1.1, np.nanmax(pivot.to_numpy(dtype=float))))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val < 0.55 else "black")
    ax.set_title("Figure 4c. SSIM retention vs ID", fontsize=10, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Retention", fontsize=8)
    fig.tight_layout()
    fig.savefig(dirs["panels"] / "fig4c_performance_retention_heatmap.png", dpi=int(cfg["plotting"]["dpi"]))
    fig.savefig(dirs["panels"] / "fig4c_performance_retention_heatmap.svg")
    plt.close(fig)


def plot_uncertainty_and_auroc(results: dict[str, pd.DataFrame], dirs: dict[str, Path], cfg: dict[str, Any]) -> None:
    metrics = results["figure_metrics"]
    failure = results["failure"]
    score_col = cfg["metrics"]["uncertainty_score"]
    conditional = cfg["models"]["conditional_model"]
    sub = metrics[metrics["model"].eq(conditional)].copy()
    if sub.empty:
        sub = metrics[metrics["model"].eq("prior_init_diffusion_unet")].copy()
    scenario_order = ["iid_test", "ood_unseen_shape", "ood_unseen_depth_range", "ood_unseen_layer_modulus", "ood_unseen_operator_noise"]
    labels = [scenario_label(s) for s in scenario_order]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
    data = [pd.to_numeric(sub.loc[sub["scenario_key"].eq(s), score_col], errors="coerce").dropna().to_numpy() for s in scenario_order]
    axes[0].violinplot([d if len(d) else np.array([np.nan]) for d in data], showmeans=True, showextrema=False)
    axes[0].set_xticks(np.arange(1, len(labels) + 1))
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[0].set_ylabel("Posterior std (normalized)", fontsize=8)
    axes[0].set_title("Uncertainty shift", fontsize=10, weight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    fail_sub = failure[failure["model"].eq(conditional)].copy()
    if fail_sub.empty:
        fail_sub = failure[failure["model"].eq("prior_init_diffusion_unet")].copy()
    fail_sub["scenario_label"] = pd.Categorical(fail_sub["scenario_label"], labels, ordered=True)
    fail_sub = fail_sub.sort_values("scenario_label")
    axes[1].bar(
        np.arange(len(fail_sub)),
        fail_sub["failure_auroc_mean"],
        yerr=fail_sub["failure_auroc_std"],
        color="#457b9d",
        edgecolor="#1f2937",
        linewidth=0.5,
    )
    axes[1].axhline(0.5, color="#6b7280", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(np.arange(len(fail_sub)))
    axes[1].set_xticklabels(fail_sub["scenario_label"], rotation=25, ha="right", fontsize=7)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Failure AUROC", fontsize=8)
    axes[1].set_title("Failure detection", fontsize=10, weight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Figure 4d. Uncertainty shift and failure detection", fontsize=11, weight="bold", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(dirs["panels"] / "fig4d_uncertainty_shift_and_auroc.png", dpi=int(cfg["plotting"]["dpi"]), bbox_inches="tight", pad_inches=0.14)
    fig.savefig(dirs["panels"] / "fig4d_uncertainty_shift_and_auroc.svg", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def plot_risk_coverage(risk: pd.DataFrame, real_anchor: pd.DataFrame, dirs: dict[str, Path], cfg: dict[str, Any]) -> None:
    if risk.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.3, 0.9]})
    method_order = ["srcnn_bz_single_channel", "unet_canonical_bxyz", "vanilla_diffusion", "explicit_residual_target_diffusion"]
    for model in method_order:
        sub = risk[risk["model"].eq(model)]
        if sub.empty:
            continue
        label = cfg["models"]["figure_methods"].get(model, model)
        axes[0].plot(sub["coverage"], sub["risk_1_minus_ssim"], marker="o", markersize=3, linewidth=1.4, label=label)
    axes[0].set_xlabel("Coverage retained", fontsize=8)
    axes[0].set_ylabel("Risk (1 - SSIM)", fontsize=8)
    axes[0].set_title("Risk-coverage curve", fontsize=10, weight="bold")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    colors = real_anchor["status"].map({"missing": "#b0b7c3"}).fillna("#e9c46a")
    axes[1].barh(np.arange(len(real_anchor)), real_anchor["found_candidate_file_count"], color=colors, edgecolor="#1f2937", linewidth=0.5)
    axes[1].set_yticks(np.arange(len(real_anchor)))
    axes[1].set_yticklabels(real_anchor["anchor_type"], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Candidate files", fontsize=8)
    axes[1].set_title("Real-world anchor audit", fontsize=10, weight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Figure 4e. Selective prediction and real-anchor availability", fontsize=11, weight="bold", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(dirs["panels"] / "fig4e_risk_coverage_and_real_anchor.png", dpi=int(cfg["plotting"]["dpi"]), bbox_inches="tight", pad_inches=0.14)
    fig.savefig(dirs["panels"] / "fig4e_risk_coverage_and_real_anchor.svg", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def downsample_to_8x8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    h, w = arr.shape[-2:]
    if h == 8 and w == 8:
        return arr.copy()
    if h % 8 == 0 and w % 8 == 0:
        return arr.reshape(8, h // 8, 8, w // 8).mean(axis=(1, 3))
    ys = np.linspace(0, h - 1, 8).astype(int)
    xs = np.linspace(0, w - 1, 8).astype(int)
    return arr[np.ix_(ys, xs)]


def pick_case_index(manifest: pd.DataFrame, case_key: str, family: str) -> int:
    if case_key == "fig4b_irregular_morphology":
        cand = manifest[(manifest["ood_split_family"].eq("ood_unseen_shape")) & (manifest["targeted_shape"].isin(["lobed", "crescent"]))]
        if cand.empty:
            cand = manifest[manifest["ood_split_family"].eq("ood_unseen_shape")]
        center = cand[["A_contrast", "H"]].median(numeric_only=True)
        score = ((cand["A_contrast"] - center["A_contrast"]).abs() / (cand["A_contrast"].std() + EPS)) + (
            (cand["H"] - center["H"]).abs() / (cand["H"].std() + EPS)
        )
        return int(score.idxmin())
    if case_key == "fig4b_deep_lesion":
        cand = manifest[manifest["ood_split_family"].eq("ood_unseen_depth_range")]
        if cand.empty:
            cand = manifest[manifest["split_role"].eq("ood_test")]
        target_depth = 5.5
        score = (cand["z_bottom"].astype(float) - target_depth).abs()
        return int(score.idxmin())
    if case_key == "fig4b_noisy_input":
        cand = manifest[(manifest["ood_split_family"].eq("ood_unseen_operator_noise")) & (manifest["operator_noise_group"].eq("high_noise"))]
        if cand.empty:
            cand = manifest[manifest["ood_split_family"].eq("ood_unseen_operator_noise")]
        score = (cand["A_contrast"].astype(float) - cand["A_contrast"].median()).abs()
        return int(score.idxmin())
    if case_key == "fig4b_missing_channels":
        cand = manifest[manifest["split_role"].eq("iid_test")]
        score = (cand["A_contrast"].astype(float) - cand["A_contrast"].median()).abs()
        return int(score.idxmin())
    cand = manifest[manifest["ood_split_family"].eq(family)]
    return int(cand.index[0])


def make_case_arrays(cfg: dict[str, Any], manifest: pd.DataFrame, cache: dict[str, np.ndarray], dirs: dict[str, Path]) -> pd.DataFrame:
    metadata_path = dirs["selected_cases"] / "fig4b_selected_cases_metadata.csv"
    if metadata_path.exists():
        existing = pd.read_csv(metadata_path)
        if not existing.empty and "npz_path" in existing:
            paths_ok = existing.loc[existing["status"].eq("generated"), "npz_path"].map(lambda p: Path(str(p)).exists()).all()
            if paths_ok:
                plot_representative_cases(existing, dirs, cfg)
                return existing

    try:
        import torch
        from torch.utils.data import DataLoader  # noqa: F401

        mn_scripts = as_path(cfg["paths"]["mn_root"]) / "scripts"
        if str(mn_scripts) not in sys.path:
            sys.path.insert(0, str(mn_scripts))
        from rspi_03_eval_posterior_metrics import load_state  # type: ignore
        from step3c_train_batch_benchmark import DiffusionDenoiser, SmallUNet  # type: ignore
        from step3j_method1_train_canonical_diffusion_cv import (  # type: ignore
            CanonicalDataset,
            NormStats,
            make_fig3_true_ood_folds,
            pick_bz_srcnn_channel,
            sample_residual_diffusion,
            set_seed,
        )
    except Exception as exc:
        error = {"representative_case_status": "import_failed", "error": repr(exc), "traceback": traceback.format_exc()}
        write_json(dirs["provenance"] / "fig4b_case_generation_error.json", error)
        return pd.DataFrame([error])

    set_seed(int(cfg["seed"]))
    run_root = as_path(cfg["paths"]["run_root"])
    seed = int(cfg["case_seed"])
    seed_dir = run_root / f"seed_{seed}" / "training" / "fig3_true_ood"
    if not seed_dir.exists():
        raise RuntimeError(f"Representative seed checkpoint dir not found: {seed_dir}")
    folds = make_fig3_true_ood_folds(manifest)
    fold_by_group = {fold["test_group"]: fold for fold in folds}
    channel_names = [str(x) for x in cache["input_channel_names"]] if "input_channel_names" in cache else []
    if not channel_names:
        channel_names = [f"channel_{i}" for i in range(cache["input64"].shape[1])]
    srcnn_channel = pick_bz_srcnn_channel(channel_names)

    device_name = cfg["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    n_samples = int(cfg["models"]["representative_posterior_samples"])
    timesteps = int(cfg["models"]["timesteps"])
    base_channels = int(cfg["models"]["base_channels"])

    rows = []
    np_cache = {
        "input64": np.asarray(cache["input64"], dtype=np.float32),
        "target_logE64": np.asarray(cache["target_logE64"], dtype=np.float32),
        "target_kpa64": np.asarray(cache["target_kpa64"], dtype=np.float32),
        "target_sigma_logE64": np.asarray(cache["target_sigma_logE64"], dtype=np.float32),
        "target_p_stiff64": np.asarray(cache["target_p_stiff64"], dtype=np.float32),
    }

    for case_key, family, title in SELECTED_CASES:
        test_group = family
        fold = fold_by_group.get(test_group)
        if fold is None:
            rows.append({"case_key": case_key, "status": "missing_fold", "scenario_key": family})
            continue
        fold_dir = seed_dir / f"fold_{int(fold['fold']):02d}"
        stats_payload = json.loads((fold_dir / "normalization_stats.json").read_text(encoding="utf-8-sig"))
        stats = NormStats(
            input_mean=np.asarray(stats_payload["input_mean"], dtype=np.float32),
            input_std=np.maximum(np.asarray(stats_payload["input_std"], dtype=np.float32), 1e-6),
            target_low=float(stats_payload["target_low_mu_logE"]),
            target_high=float(stats_payload["target_high_mu_logE"]),
            target_mean_norm=float(stats_payload["target_mean_norm"]),
        )
        idx = pick_case_index(manifest, case_key, family)
        ds = CanonicalDataset(manifest, np_cache, [idx], stats, srcnn_channel, augment=False)
        item = ds[0]
        batch: dict[str, Any] = {}
        for key, value in item.items():
            if hasattr(value, "unsqueeze"):
                batch[key] = value.unsqueeze(0)
            else:
                batch[key] = [value]

        observed_mask_8 = np.ones((8, 8), dtype=np.float32)
        synthetic_note = ""
        if case_key == "fig4b_missing_channels":
            x = batch["input"].clone()
            x[:, :, 16:24, :] = 0.0
            x[:, :, :, 40:48] = 0.0
            batch["input"] = x
            observed_mask_8[2, :] = 0.0
            observed_mask_8[:, 5] = 0.0
            synthetic_note = "Controlled missing-channel perturbation applied to an ID held-out case."

        unet = SmallUNet(np_cache["input64"].shape[1], out_ch=1, base=base_channels)
        unet.load_state_dict(load_state(fold_dir / "unet_best.pt"))
        unet.to(device)
        residual_checkpoint = torch.load(fold_dir / "explicit_residual_target_diffusion.pt", map_location="cpu", weights_only=False)
        residual_scale = float(residual_checkpoint.get("residual_scale", 0.25))
        residual_diffusion = DiffusionDenoiser(cond_ch=np_cache["input64"].shape[1], timesteps=timesteps, base=base_channels)
        residual_diffusion.load_state_dict(residual_checkpoint["model"])
        residual_diffusion.to(device)
        with torch.no_grad():
            samples = sample_residual_diffusion(
                residual_diffusion,
                unet,
                batch,
                device,
                timesteps,
                residual_scale,
                n_samples,
                surrogate=None,
                guidance_scale=0.0,
            )
        samples_np = samples.numpy()
        mean_norm = samples_np.mean(axis=0).squeeze()
        std_norm = samples_np.std(axis=0).squeeze()
        ref_kpa = np_cache["target_kpa64"][idx]
        recon_log = mean_norm * max(stats.target_high - stats.target_low, EPS) + stats.target_low
        recon_kpa = np.exp(recon_log)
        std_log = std_norm * max(stats.target_high - stats.target_low, EPS)
        uncertainty_kpa = recon_kpa * std_log
        b8 = downsample_to_8x8(np_cache["input64"][idx, srcnn_channel])
        if case_key == "fig4b_missing_channels":
            b8 = b8.copy()
            b8[observed_mask_8 == 0] = np.nan

        out_npz = dirs["selected_cases"] / f"{case_key}.npz"
        np.savez_compressed(
            out_npz,
            B8=b8.astype(np.float32),
            reference_E_kPa=ref_kpa.astype(np.float32),
            recon_E_kPa=recon_kpa.astype(np.float32),
            uncertainty_kPa=uncertainty_kpa.astype(np.float32),
            uncertainty_norm=std_norm.astype(np.float32),
            lesion_mask=np_cache["target_p_stiff64"][idx].astype(np.float32),
            observed_mask_8x8=observed_mask_8.astype(np.float32),
        )
        prefix = dirs["arrays"] / case_key
        np.save(f"{prefix}_B8.npy", b8.astype(np.float32))
        np.save(f"{prefix}_reference_E.npy", ref_kpa.astype(np.float32))
        np.save(f"{prefix}_recon_E.npy", recon_kpa.astype(np.float32))
        np.save(f"{prefix}_uncertainty.npy", uncertainty_kpa.astype(np.float32))

        rows.append(
            {
                "case_key": case_key,
                "display_title": title,
                "status": "generated",
                "scenario_key": family,
                "scenario_label": scenario_label(family),
                "sample_index": int(idx),
                "condition_id": str(manifest.loc[idx, "condition_id"]),
                "fold": int(fold["fold"]),
                "seed": seed,
                "checkpoint_dir": str(fold_dir),
                "posterior_samples_for_case": n_samples,
                "srcnn_channel": int(srcnn_channel),
                "srcnn_channel_name": channel_names[srcnn_channel],
                "residual_scale": residual_scale,
                "synthetic_note": synthetic_note,
                "npz_path": str(out_npz),
            }
        )
    cases = pd.DataFrame(rows)
    cases.to_csv(dirs["selected_cases"] / "fig4b_selected_cases_metadata.csv", index=False)
    plot_representative_cases(cases, dirs, cfg)
    return cases


def plot_representative_cases(cases: pd.DataFrame, dirs: dict[str, Path], cfg: dict[str, Any]) -> None:
    generated = cases[cases["status"].eq("generated")]
    if generated.empty:
        return
    fig, axes = plt.subplots(len(generated), 4, figsize=(8.2, 2.0 * len(generated)))
    if len(generated) == 1:
        axes = np.asarray([axes])
    col_titles = ["8x8 input", "Reference E", "Conditional diffusion", "Uncertainty"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=8, weight="bold")
    for r, (_, row) in enumerate(generated.iterrows()):
        data = np.load(row["npz_path"])
        b8 = data["B8"]
        ref = data["reference_E_kPa"]
        recon = data["recon_E_kPa"]
        unc = data["uncertainty_kPa"]
        vmax = np.nanpercentile(np.r_[ref.ravel(), recon.ravel()], 99)
        vmin = np.nanpercentile(np.r_[ref.ravel(), recon.ravel()], 1)
        axes[r, 0].imshow(b8, cmap="coolwarm")
        axes[r, 1].imshow(ref, cmap=cfg["plotting"]["colormap_stiffness"], vmin=vmin, vmax=vmax)
        axes[r, 2].imshow(recon, cmap=cfg["plotting"]["colormap_stiffness"], vmin=vmin, vmax=vmax)
        axes[r, 3].imshow(unc, cmap=cfg["plotting"]["colormap_uncertainty"])
        axes[r, 0].set_ylabel(row["display_title"], fontsize=8)
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle("Figure 4b. Representative OOD cases", fontsize=11, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(dirs["panels"] / "fig4b_representative_ood_cases.png", dpi=int(cfg["plotting"]["dpi"]), bbox_inches="tight", pad_inches=0.14)
    fig.savefig(dirs["panels"] / "fig4b_representative_ood_cases.svg", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def make_full_preview(dirs: dict[str, Path], cfg: dict[str, Any]) -> None:
    image_names = [
        "fig4a_ood_taxonomy.png",
        "fig4b_representative_ood_cases.png",
        "fig4c_performance_retention_heatmap.png",
        "fig4d_uncertainty_shift_and_auroc.png",
        "fig4e_risk_coverage_and_real_anchor.png",
    ]
    paths = [dirs["panels"] / name for name in image_names if (dirs["panels"] / name).exists()]
    if not paths:
        return
    fig, axes = plt.subplots(len(paths), 1, figsize=(9.5, 3.2 * len(paths)))
    if len(paths) == 1:
        axes = [axes]
    for ax, path in zip(axes, paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(dirs["panels"] / "fig4_full_preview.png", dpi=200)
    plt.close(fig)


def write_readme(
    cfg: dict[str, Any],
    manifest: pd.DataFrame,
    taxonomy: pd.DataFrame,
    seed_status: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    real_anchor: pd.DataFrame,
    cases: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    complete_seeds = seed_status.loc[seed_status["status"].eq("complete"), "seed"].tolist()
    missing_seeds = seed_status.loc[~seed_status["status"].eq("complete"), "seed"].tolist()
    cond = cfg["models"]["conditional_model"]
    perf = results["performance"]
    cond_perf = perf[perf["model"].eq(cond)][
        ["scenario_label", "ssim_norm_mean", "mae_kpa_mean", "posterior_std_norm_mean_mean", "failure_auroc_top10_abs_error_mean"]
    ].copy()
    readme = [
        "# Figure 4 OOD Robustness Data Package",
        "",
        "This folder processes the available full512 Method 2b COMSOL/diffusion OOD outputs for the Advanced Science Figure 4 plan.",
        "",
        "## Source data",
        f"- Manifest rows: {len(manifest)}",
        f"- Run root: `{cfg['paths']['run_root']}`",
        f"- Cache dir: `{cfg['paths']['cache_dir']}`",
        f"- Complete seeds used: {complete_seeds}",
        f"- Missing/incomplete planned seeds: {missing_seeds}",
        "- Existing OOD families: morphology/shape, depth range, layer modulus, and operator noise.",
        "- Size, compression-mismatch, and full missing-channel metric sets were not present in the current full512 run; missing-channel is included only as a controlled representative-case perturbation.",
        "",
        "## Outputs",
        "- `metrics/id_ood_split_summary.csv`: ID/OOD split counts.",
        "- `metrics/fig4_fold_level_metrics.csv`: seed x model x scenario metrics copied from the completed posterior evaluator.",
        "- `metrics/fig4_performance_by_method_scenario.csv`: cross-seed summaries.",
        "- `metrics/fig4_performance_retention.csv`: OOD/ID retention ratios.",
        "- `metrics/fig4_uncertainty_shift.csv`: group-level uncertainty shift tests and BH-adjusted q values.",
        "- `metrics/fig4_failure_detection_auroc.csv`: existing pixel-error failure AUROC summaries.",
        "- `metrics/fig4_risk_coverage.csv`: group-level risk-coverage curves.",
        "- `selected_cases/*.npz` and `arrays/*.npy`: Figure 4b representative case arrays.",
        "- `panels/*.png` and `panels/*.svg`: Figure 4a-e preview panels.",
        "",
        "## Conditional diffusion summary",
        markdown_table(cond_perf.round(4)) if not cond_perf.empty else "Conditional diffusion rows were not found.",
        "",
        "## Real-anchor audit",
        markdown_table(real_anchor[["anchor_type", "found_candidate_file_count", "status"]]),
        "",
        "Candidate real-anchor files are not automatically treated as usable unless they match the B8/reference-map schema. No synthetic real-world anchor data were fabricated.",
        "",
        "## Representative cases",
        markdown_table(cases[["case_key", "status", "scenario_label", "condition_id", "synthetic_note"]])
        if not cases.empty and "condition_id" in cases
        else "Representative cases were not generated.",
        "",
        "## Limitations",
        "- The completed posterior evaluator stores fold/test-group metrics, not per-sample posterior arrays for all 512 cases. Therefore uncertainty shift and risk-coverage are reported at seed-test-group granularity.",
        "- Figure 4b arrays are generated from the selected seed checkpoint only and are intended as representative visual material, while quantitative panels use all complete seeds.",
        "- Current full512 outputs use 64x64 target maps. The pipeline keeps native resolution instead of upsampling to 256x256.",
        "",
    ]
    write_text(dirs["root"] / "README_FIG4.md", "\n".join(readme))

    methods = [
        "Figure 4 OOD analysis reused the full512 Method 2b COMSOL/diffusion cache and completed seed-level posterior metrics.",
        "ID samples were held-out iid_support cases, while OOD samples comprised unseen morphology, deeper depth range, unseen layer-modulus variation, and operator-noise groups.",
        "Performance retention was computed as OOD/ID for higher-better metrics and ID/OOD for lower-better errors.",
        "Uncertainty shift and risk-coverage used posterior_std_norm_mean at the seed-test-group level because full per-sample posterior arrays were not stored by the completed evaluator.",
        "Representative cases were regenerated from existing checkpoints; missing-channel visualization used a controlled sensor-mask perturbation of an ID held-out case and was not included as a full metric family.",
    ]
    write_text(dirs["root"] / "fig4_methods_snippet.txt", "\n".join(methods) + "\n")


def run(config_path: Path) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    out_dir = as_path(cfg["paths"]["output_dir"])
    dirs = ensure_dirs(out_dir)
    manifest, cache, cache_summary = load_source_data(cfg)
    metrics, seed_status = load_seed_metrics(cfg, dirs)
    taxonomy = write_manifest_outputs(cfg, manifest, cache, cache_summary, seed_status, dirs)
    real_anchor = audit_real_anchors(cfg, dirs)
    results = summarize_metrics(cfg, metrics, dirs)

    plot_taxonomy(taxonomy, dirs, cfg)
    plot_retention(results["retention"], dirs, cfg)
    plot_uncertainty_and_auroc(results, dirs, cfg)
    plot_risk_coverage(results["risk_coverage"], real_anchor, dirs, cfg)

    try:
        cases = make_case_arrays(cfg, manifest, cache, dirs)
    except Exception as exc:
        cases = pd.DataFrame(
            [{"case_key": "all", "status": "generation_failed", "error": repr(exc), "traceback": traceback.format_exc()}]
        )
        cases.to_csv(dirs["selected_cases"] / "fig4b_selected_cases_metadata.csv", index=False)
        write_json(dirs["provenance"] / "fig4b_case_generation_error.json", cases.iloc[0].to_dict())

    make_full_preview(dirs, cfg)
    write_readme(cfg, manifest, taxonomy, seed_status, results, real_anchor, cases, dirs)

    summary = {
        "status": "completed",
        "output_dir": str(out_dir),
        "manifest_rows": int(len(manifest)),
        "complete_seeds": seed_status.loc[seed_status["status"].eq("complete"), "seed"].astype(int).tolist(),
        "missing_or_incomplete_seeds": seed_status.loc[~seed_status["status"].eq("complete"), "seed"].astype(int).tolist(),
        "panel_dir": str(dirs["panels"]),
        "metrics_dir": str(dirs["metrics"]),
        "selected_cases_generated": int((cases.get("status", pd.Series(dtype=str)) == "generated").sum()) if not cases.empty else 0,
    }
    write_json(dirs["provenance"] / "fig4_processing_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Process Figure 4 OOD robustness outputs.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "fig4_ood_config.yaml")
    args = parser.parse_args()
    summary = run(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
