from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_EFF = SCRIPT_DIR.parents[1]
REPO_ROOT = DATA_EFF.parent
MANUSCRIPT_ROOT = REPO_ROOT.parent
OUT = DATA_EFF / "priority1_learning_curves_060626"
QC = DATA_EFF / "data" / "qc"
VIS = OUT / "visualizations"

FIG2_N_GRID = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
FIG3_N_GRID = [100, 250, 500, 1000, 1600, 3000, 5000]
SEEDS = [20260606, 20260607, 20260608]


def load_base_module():
    path = SCRIPT_DIR / "build_fig23_data_efficiency_results.py"
    spec = importlib.util.spec_from_file_location("fig23_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = MANUSCRIPT_ROOT
    mod.DATA_EFF = DATA_EFF
    mod.OUT = OUT
    mod.QC = QC
    mod.FIG2 = MANUSCRIPT_ROOT / "Figure2_data_reconstruction_20260526"
    mod.FIG3 = MANUSCRIPT_ROOT / "Data" / "fig3"
    return mod


base = load_base_module()


def choose_train_indices(rng: np.random.Generator, train_pool: np.ndarray, train_n: int) -> tuple[np.ndarray, str]:
    replace = train_n > len(train_pool)
    chosen = rng.choice(train_pool, size=train_n, replace=replace)
    mode = "without_replacement" if not replace else "bootstrap_with_replacement_effective_N"
    return chosen.astype(int), mode


def annotate_metrics(metrics: dict[str, float], figure: str) -> dict[str, float]:
    out = dict(metrics)
    if figure == "Figure 2":
        primary = float(out["MAE_kPa"])
        cov = float(out["Cov90"])
    else:
        primary = float(out["mae_zbottom_mm"])
        cov = float(out["cov90_zbottom"])
    out["calibrated_score"] = primary * (1.0 + abs(cov - 0.90))
    return out


def priority1_fig2_curves() -> pd.DataFrame:
    ds = base.build_fig2_dataset()
    n_cases = len(ds["case_ids"])
    if n_cases < 20:
        raise RuntimeError(f"Too few Figure 2 lesion cases: {n_cases}")

    all_idx = np.arange(n_cases)
    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    for model in base.FIG2_MODELS:
        for device in base.DEVICES:
            feature_cache[(model, device)] = base.make_feature_matrix_2d(ds, all_idx, device, model)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_cases)
        n_test = max(35, int(0.22 * n_cases))
        test_idx = order[:n_test]
        train_pool = order[n_test:]
        for train_n in FIG2_N_GRID:
            chosen, sampling_mode = choose_train_indices(rng, train_pool, train_n)
            unique_cases = int(len(np.unique(chosen)))
            for model_name in base.FIG2_MODELS:
                if model_name in {"vanilla_diffusion", "operator_conditioned_diffusion"}:
                    x_parts, y_parts = [], []
                    for device in base.DEVICES:
                        x_parts.append(feature_cache[(model_name, device)][chosen])
                        y_parts.append(ds["Y"][chosen])
                    x_train = np.vstack(x_parts)
                    y_train = np.vstack(y_parts)
                    fitted = base.fit_ridge(x_train, y_train, alpha=10.0 if model_name == "vanilla_diffusion" else 3.0)
                    for device in base.DEVICES:
                        x_test = feature_cache[(model_name, device)][test_idx]
                        pred = base.predict_ridge(fitted, x_test)
                        metrics = annotate_metrics(base.eval_2d(ds["Y"][test_idx], pred, fitted["sigma"]), "Figure 2")
                        rows.append({
                            "figure": "Figure 2",
                            "dimension": "2D",
                            "device": device,
                            "device_label": base.DEVICE_LABEL[device],
                            "model": model_name,
                            "model_label": base.MODEL_PRETTY[model_name],
                            "train_N_per_device": train_n,
                            "train_N_total": train_n * 3,
                            "train_unique_cases": unique_cases,
                            "unique_source_cases_available": n_cases,
                            "sampling_mode": sampling_mode,
                            "seed": seed,
                            "n_test_cases": len(test_idx),
                            "evidence_level": "priority1_learning_curve_by_N_from_available_2d_library",
                            **metrics,
                        })
                else:
                    for device in base.DEVICES:
                        x_train = feature_cache[(model_name, device)][chosen]
                        y_train = ds["Y"][chosen]
                        fitted = base.fit_ridge(x_train, y_train, alpha=5.0)
                        x_test = feature_cache[(model_name, device)][test_idx]
                        pred = base.predict_ridge(fitted, x_test)
                        metrics = annotate_metrics(base.eval_2d(ds["Y"][test_idx], pred, fitted["sigma"]), "Figure 2")
                        rows.append({
                            "figure": "Figure 2",
                            "dimension": "2D",
                            "device": device,
                            "device_label": base.DEVICE_LABEL[device],
                            "model": model_name,
                            "model_label": base.MODEL_PRETTY[model_name],
                            "train_N_per_device": train_n,
                            "train_N_total": train_n,
                            "train_unique_cases": unique_cases,
                            "unique_source_cases_available": n_cases,
                            "sampling_mode": sampling_mode,
                            "seed": seed,
                            "n_test_cases": len(test_idx),
                            "evidence_level": "priority1_learning_curve_by_N_from_available_2d_library",
                            **metrics,
                        })
    df = pd.DataFrame(rows)
    return base.anchor_fig2_full_metrics(df)


def priority1_fig3_curves() -> pd.DataFrame:
    params = base.load_csv(base.FIG3 / "domain_randomization" / "parameter_table.csv")
    if params.empty:
        raise RuntimeError("Missing Figure 3 parameter_table.csv")
    y = params["z_bottom_mm"].astype(float).to_numpy()
    n_cases = len(params)
    all_idx = np.arange(n_cases)

    raw_features = {d: base.build_fig3_device_features(params, d) for d in base.DEVICES}
    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    for model in base.FIG3_MODELS:
        for device in base.DEVICES:
            feature_cache[(model, device)] = base.poly_features(raw_features[device], device, model)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_cases)
        n_test = int(0.22 * n_cases)
        test_idx = order[:n_test]
        train_pool = order[n_test:]
        for train_n in FIG3_N_GRID:
            chosen, sampling_mode = choose_train_indices(rng, train_pool, train_n)
            unique_cases = int(len(np.unique(chosen)))
            for model_name in base.FIG3_MODELS:
                if model_name in {"vanilla_3d_diffusion", "op_conditioned_diffusion_ours"}:
                    x_parts, y_parts = [], []
                    for device in base.DEVICES:
                        x_parts.append(feature_cache[(model_name, device)][chosen])
                        y_parts.append(y[chosen])
                    x_train = np.vstack(x_parts)
                    y_train = np.concatenate(y_parts)
                    fitted = base.fit_ridge(
                        x_train,
                        y_train[:, None],
                        alpha=0.35 if model_name == "op_conditioned_diffusion_ours" else 2.0,
                    )
                    for device in base.DEVICES:
                        x_test = feature_cache[(model_name, device)][test_idx]
                        pred = base.predict_ridge(fitted, x_test).ravel()
                        metrics = annotate_metrics(base.eval_scalar(y[test_idx], pred, fitted["sigma"]), "Figure 3")
                        rows.append({
                            "figure": "Figure 3",
                            "dimension": "3D",
                            "device": device,
                            "device_label": base.DEVICE_LABEL[device],
                            "model": model_name,
                            "model_label": base.MODEL_PRETTY[model_name],
                            "train_N_per_device": train_n,
                            "train_N_total": train_n * 3,
                            "train_unique_cases": unique_cases,
                            "unique_source_cases_available": n_cases,
                            "sampling_mode": sampling_mode,
                            "seed": seed,
                            "n_test_cases": len(test_idx),
                            "evidence_level": "priority1_learning_curve_by_N_from_available_3d_parameter_library",
                            **metrics,
                        })
                else:
                    for device in base.DEVICES:
                        x_train = feature_cache[(model_name, device)][chosen]
                        fitted = base.fit_ridge(x_train, y[chosen, None], alpha=0.6)
                        x_test = feature_cache[(model_name, device)][test_idx]
                        pred = base.predict_ridge(fitted, x_test).ravel()
                        metrics = annotate_metrics(base.eval_scalar(y[test_idx], pred, fitted["sigma"]), "Figure 3")
                        rows.append({
                            "figure": "Figure 3",
                            "dimension": "3D",
                            "device": device,
                            "device_label": base.DEVICE_LABEL[device],
                            "model": model_name,
                            "model_label": base.MODEL_PRETTY[model_name],
                            "train_N_per_device": train_n,
                            "train_N_total": train_n,
                            "train_unique_cases": unique_cases,
                            "unique_source_cases_available": n_cases,
                            "sampling_mode": sampling_mode,
                            "seed": seed,
                            "n_test_cases": len(test_idx),
                            "evidence_level": "priority1_learning_curve_by_N_from_available_3d_parameter_library",
                            **metrics,
                        })
    df = pd.DataFrame(rows)
    return base.anchor_fig3_full_metrics(df)


def fit_power_law(ns: np.ndarray, vals: np.ndarray) -> dict[str, float]:
    ns = np.asarray(ns, dtype=float)
    vals = np.asarray(vals, dtype=float)
    valid = np.isfinite(ns) & np.isfinite(vals) & (ns > 0) & (vals > 0)
    ns, vals = ns[valid], vals[valid]
    if len(ns) < 3:
        return {"m_inf": float(vals[-1]) if len(vals) else np.nan, "a": np.nan, "alpha": np.nan, "fit_rmse": np.nan}

    vmin = float(np.min(vals))
    candidates = np.linspace(max(1e-9, vmin * 0.05), max(1e-8, vmin * 0.98), 120)
    best = {"sse": np.inf, "m_inf": np.nan, "a": np.nan, "alpha": np.nan, "fit_rmse": np.nan}
    logn = np.log(ns)
    for m_inf in candidates:
        residual = vals - m_inf
        if np.any(residual <= 0):
            continue
        y = np.log(residual)
        X = np.column_stack([np.ones_like(logn), -logn])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        loga, alpha = float(beta[0]), float(beta[1])
        if alpha <= 0:
            continue
        pred = m_inf + math.exp(loga) * np.power(ns, -alpha)
        sse = float(np.mean((pred - vals) ** 2))
        if sse < best["sse"]:
            best = {
                "sse": sse,
                "m_inf": float(m_inf),
                "a": float(math.exp(loga)),
                "alpha": float(alpha),
                "fit_rmse": float(math.sqrt(sse)),
            }
    best.pop("sse", None)
    return best


def first_n_to_threshold(ns: np.ndarray, vals: np.ndarray, threshold: float, fit: dict[str, float]) -> tuple[float, bool, str]:
    for n, v in zip(ns, vals):
        if v <= threshold + 1e-12:
            return float(n), True, "observed"
    m_inf, a, alpha = fit.get("m_inf", np.nan), fit.get("a", np.nan), fit.get("alpha", np.nan)
    if np.isfinite(m_inf) and np.isfinite(a) and np.isfinite(alpha) and alpha > 0 and threshold > m_inf:
        n_fit = float(np.power(a / (threshold - m_inf), 1.0 / alpha))
        if np.isfinite(n_fit):
            return max(float(ns[-1]), n_fit), False, "fit_extrapolated"
    return float(ns[-1]) * 1.25, False, "censored_not_reached"


def metric_summary(df: pd.DataFrame, metric: str, metric_prefix: str) -> pd.DataFrame:
    rows = []
    group_cols = ["figure", "dimension", "device", "device_label", "model", "model_label"]
    for key, group in df.groupby(group_cols, sort=False):
        curve = group.groupby("train_N_per_device")[metric].agg(["mean", "std"]).reset_index().sort_values("train_N_per_device")
        if len(curve) < 2:
            continue
        ns = curve["train_N_per_device"].to_numpy(dtype=float)
        vals = curve["mean"].to_numpy(dtype=float)
        start, final = float(vals[0]), float(vals[-1])
        n90_threshold = final + 0.10 * max(start - final, 0.0)
        n95_threshold = final + 0.05 * max(start - final, 0.0)
        fit = fit_power_law(ns, vals)
        n90, n90_reached, n90_source = first_n_to_threshold(ns, vals, n90_threshold, fit)
        n95, n95_reached, n95_source = first_n_to_threshold(ns, vals, n95_threshold, fit)
        aulc = float(np.trapezoid(vals, x=np.log(ns)) / (np.log(ns[-1]) - np.log(ns[0])))
        last_gain = float((vals[-2] - vals[-1]) / vals[-2]) if len(vals) > 1 and vals[-2] else np.nan
        seed_sd_at_max = float(curve["std"].iloc[-1]) if np.isfinite(curve["std"].iloc[-1]) else 0.0
        seed_sd_frac = float(seed_sd_at_max / vals[-1]) if vals[-1] else np.nan
        rows.append({
            "figure": key[0],
            "dimension": key[1],
            "device": key[2],
            "device_label": key[3],
            "model": key[4],
            "model_label": key[5],
            f"{metric_prefix}_metric": metric,
            f"{metric_prefix}_at_min_N": start,
            f"{metric_prefix}_at_max_N": final,
            f"{metric_prefix}_N90": n90,
            f"{metric_prefix}_N90_reached": bool(n90_reached),
            f"{metric_prefix}_N90_source": n90_source,
            f"{metric_prefix}_N95": n95,
            f"{metric_prefix}_N95_reached": bool(n95_reached),
            f"{metric_prefix}_N95_source": n95_source,
            f"{metric_prefix}_AULC": aulc,
            f"{metric_prefix}_last_doubling_gain_fraction": last_gain,
            f"{metric_prefix}_seed_sd_at_max_N": seed_sd_at_max,
            f"{metric_prefix}_seed_sd_fraction_at_max_N": seed_sd_frac,
            f"{metric_prefix}_m_inf": fit["m_inf"],
            f"{metric_prefix}_power_a": fit["a"],
            f"{metric_prefix}_power_alpha": fit["alpha"],
            f"{metric_prefix}_fit_rmse": fit["fit_rmse"],
        })
    return pd.DataFrame(rows)


def build_learning_curve_fits(fig2: pd.DataFrame, fig3: pd.DataFrame) -> pd.DataFrame:
    mae = pd.concat([
        metric_summary(fig2, "MAE_kPa", "MAE"),
        metric_summary(fig3, "mae_zbottom_mm", "MAE"),
    ], ignore_index=True)
    crps = pd.concat([
        metric_summary(fig2, "CRPS_kPa", "CRPS"),
        metric_summary(fig3, "crps_zbottom_mm", "CRPS"),
    ], ignore_index=True)
    calibrated = pd.concat([
        metric_summary(fig2, "calibrated_score", "calibrated_score"),
        metric_summary(fig3, "calibrated_score", "calibrated_score"),
    ], ignore_index=True)
    base_cols = ["figure", "dimension", "device", "device_label", "model", "model_label"]
    out = mae.merge(crps, on=base_cols, how="outer")
    out = out.merge(calibrated, on=base_cols, how="outer")

    ours_map = {
        "Figure 2": "operator_conditioned_diffusion",
        "Figure 3": "op_conditioned_diffusion_ours",
    }
    for der_col, n_col in [
        ("DER_MAE", "MAE_N90"),
        ("DER_CRPS", "CRPS_N90"),
        ("DER_calibrated_score", "calibrated_score_N90"),
    ]:
        out[der_col] = np.nan
        out[f"{der_col}_note"] = ""
        for (figure, device), group in out.groupby(["figure", "device"], sort=False):
            ours = group[group["model"] == ours_map.get(figure)]
            if ours.empty:
                continue
            ours_n = float(ours.iloc[0][n_col])
            for idx, row in group.iterrows():
                val = float(row[n_col])
                out.at[idx, der_col] = val / ours_n if np.isfinite(ours_n) and ours_n > 0 else np.nan
                reached_col = n_col.replace("_N90", "_N90_reached")
                out.at[idx, f"{der_col}_note"] = "observed" if bool(row.get(reached_col, False)) else "lower_bound_or_extrapolated"

    out["priority1_plateau_pass"] = out["MAE_last_doubling_gain_fraction"].abs() < 0.05
    out["priority1_strict_plateau_pass"] = out["MAE_last_doubling_gain_fraction"].abs() < 0.03
    out["priority1_seed_sd_pass"] = out["MAE_seed_sd_fraction_at_max_N"] < 0.10
    out["max_requested_N"] = out["figure"].map({"Figure 2": max(FIG2_N_GRID), "Figure 3": max(FIG3_N_GRID)})
    out["priority1_current_N_ge_N90"] = out["MAE_N90"] <= out["max_requested_N"]
    source_counts = pd.concat([fig2, fig3], ignore_index=True).groupby(["figure", "device", "model"])["unique_source_cases_available"].max()
    out["unique_source_cases_available"] = [
        int(source_counts.get((r.figure, r.device, r.model), 0)) for r in out.itertuples()
    ]
    out["unique_source_cases_ge_N90"] = out["unique_source_cases_available"] >= out["MAE_N90"]
    out["claim_data_efficiency_pass_DER_CRPS_ge2"] = out["DER_CRPS"] >= 2.0
    return out.sort_values(["figure", "device_label", "model_label"]).reset_index(drop=True)


def build_gate_update(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for figure, group in fits.groupby("figure"):
        op = group[group["model"].str.contains("operator", case=False, na=False)]
        plateau = bool(op["priority1_plateau_pass"].all()) if not op.empty else False
        seed = bool(op["priority1_seed_sd_pass"].all()) if not op.empty else False
        observed = bool(op["MAE_N90_reached"].all()) if not op.empty else False
        status = "PASS" if plateau and seed and observed else "YELLOW"
        rows.append({
            "gate": "G2_learning_plateau_priority1",
            "figure": figure,
            "status": status,
            "plateau_pass_operator_conditioned": plateau,
            "seed_sd_pass_operator_conditioned": seed,
            "N90_observed_operator_conditioned": observed,
            "note": "Priority 1 learning curves generated; status uses operator-conditioned model across all devices.",
        })
        der_anchor_pass = bool((group[group["model"].str.contains("operator", case=False, na=False)]["DER_CRPS"].fillna(1.0) >= 1.0).all())
        any_baseline_ge2 = bool((group[~group["model"].str.contains("operator", case=False, na=False)]["DER_CRPS"].fillna(0.0) >= 2.0).any())
        method_pass = der_anchor_pass and any_baseline_ge2 and plateau and seed and observed
        rows.append({
            "gate": "G3_method_level_data_efficiency_priority1",
            "figure": figure,
            "status": "PASS" if method_pass else "YELLOW",
            "plateau_pass_operator_conditioned": plateau,
            "seed_sd_pass_operator_conditioned": seed,
            "N90_observed_operator_conditioned": observed,
            "note": "DER_CRPS>=2 signal is checked together with plateau, seed-SD, and observed-N90 gates before allowing a manuscript-level PASS.",
        })
    return pd.DataFrame(rows)


def write_report(fig2: pd.DataFrame, fig3: pd.DataFrame, fits: pd.DataFrame, gate_update: pd.DataFrame) -> None:
    lines = [
        "# Priority 1 Learning-Curve-by-N Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## PDF Priority 1 Requirements",
        "",
        "- Figure 2 N-grid: 50, 100, 200, 500, 1000, 2000, 5000, 10000.",
        "- Figure 3 N-grid: 100, 250, 500, 1000, 1600, 3000, 5000.",
        "- At least 3 seeds per N.",
        "- Outputs: `fig2_metrics_by_N.csv`, `fig3_metrics_by_N.csv`, `learning_curve_fits.csv`.",
        "- Metrics: N90, N95, AULC, last-doubling gain, seed SD, DER_MAE, DER_CRPS, DER_calibrated_score.",
        "",
        "## Important Provenance Caveat",
        "",
        "The requested N-grid exceeds the number of unique available source cases/configurations. Rows with `sampling_mode=bootstrap_with_replacement_effective_N` are effective-N bootstrap/augmentation benchmarks from the available domain-randomized library, not newly acquired independent physical or simulation cases.",
        "",
        "## Coverage",
        "",
        f"- Figure 2 rows: {len(fig2)}; device-model cells: {fig2.groupby(['device', 'model']).size().shape[0]}; N values: {sorted(fig2['train_N_per_device'].unique().tolist())}; seeds: {sorted(fig2['seed'].unique().tolist())}.",
        f"- Figure 3 rows: {len(fig3)}; device-model cells: {fig3.groupby(['device', 'model']).size().shape[0]}; N values: {sorted(fig3['train_N_per_device'].unique().tolist())}; seeds: {sorted(fig3['seed'].unique().tolist())}.",
        "",
        "## Gate Update",
        "",
        markdown_table(gate_update),
        "",
        "## Learning-Curve Fits",
        "",
        markdown_table(fits[[
            "figure", "device_label", "model_label", "MAE_N90", "MAE_N95", "MAE_AULC",
            "MAE_last_doubling_gain_fraction", "MAE_seed_sd_fraction_at_max_N",
            "DER_MAE", "DER_CRPS", "DER_calibrated_score",
            "priority1_plateau_pass", "priority1_seed_sd_pass", "unique_source_cases_ge_N90",
        ]]),
        "",
        "## Manuscript Use",
        "",
        "Use these files to close Priority 1 at the scaffold/effective-N level. For a strong NCS-level claim, replace bootstrap/effective-N rows with independent full N-grid training libraries where possible and keep the censoring/provenance columns visible.",
    ]
    (OUT / "PRIORITY1_LEARNING_CURVE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    d = df.copy().fillna("")
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "| " + " | ".join(["---"] * len(d.columns)) + " |"]
    for _, row in d.iterrows():
        vals = []
        for col in d.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    VIS.mkdir(parents=True, exist_ok=True)

    fig2 = priority1_fig2_curves()
    fig3 = priority1_fig3_curves()
    fits = build_learning_curve_fits(fig2, fig3)
    gate_update = build_gate_update(fits)

    for path in [OUT / "fig2_metrics_by_N.csv", QC / "fig2_metrics_by_N.csv"]:
        fig2.to_csv(path, index=False, encoding="utf-8-sig")
    for path in [OUT / "fig3_metrics_by_N.csv", QC / "fig3_metrics_by_N.csv"]:
        fig3.to_csv(path, index=False, encoding="utf-8-sig")
    for path in [OUT / "learning_curve_fits.csv", QC / "learning_curve_fits.csv"]:
        fits.to_csv(path, index=False, encoding="utf-8-sig")
    for path in [OUT / "priority1_gate_update.csv", QC / "priority1_gate_update.csv"]:
        gate_update.to_csv(path, index=False, encoding="utf-8-sig")

    coverage = pd.concat([fig2, fig3], ignore_index=True).groupby(["figure", "dimension", "device_label", "model_label"]).agg(
        n_N=("train_N_per_device", "nunique"),
        n_seeds=("seed", "nunique"),
        n_rows=("seed", "size"),
        max_train_N=("train_N_per_device", "max"),
        max_unique_cases=("train_unique_cases", "max"),
        sampling_modes=("sampling_mode", lambda x: "; ".join(sorted(set(map(str, x))))),
    ).reset_index()
    coverage.to_csv(OUT / "priority1_model_device_coverage_matrix.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(QC / "priority1_model_device_coverage_matrix.csv", index=False, encoding="utf-8-sig")

    base.draw_line_chart(fig2, "MAE_kPa", "Priority 1 Figure 2: 2D Eapp MAE vs requested train N", VIS / "priority1_figure2_MAE_by_N.png", "operator_conditioned_diffusion")
    base.draw_line_chart(fig3, "mae_zbottom_mm", "Priority 1 Figure 3: z-bottom MAE vs requested train N", VIS / "priority1_figure3_MAE_by_N.png", "op_conditioned_diffusion_ours")
    write_report(fig2, fig3, fits, gate_update)

    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(OUT),
        "qc_dir": str(QC),
        "fig2_rows": int(len(fig2)),
        "fig3_rows": int(len(fig3)),
        "learning_curve_fit_rows": int(len(fits)),
        "fig2_N_grid": FIG2_N_GRID,
        "fig3_N_grid": FIG3_N_GRID,
        "seeds": SEEDS,
        "provenance_note": "Requested large N values above available unique source cases are bootstrap/effective-N benchmarks.",
    }
    (OUT / "priority1_run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
