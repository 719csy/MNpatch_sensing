from __future__ import annotations

import json
import math
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"H:\My Drive\Manuscript\Sparse reconstruction")
DATA_EFF = ROOT / "Data" / "Data efficiency"
PREV_OUT = DATA_EFF / "fig23_data_efficiency_outputs"
OUT = DATA_EFF / "ncs_audit_20260604"
FIG2 = ROOT / "Figure2_data_reconstruction_20260526"
FIG3_DATA = ROOT / "Data" / "fig3"
FIG3_FIG = ROOT / "figures" / "figure3"


COLORS = {
    "PASS": (33, 126, 84),
    "YELLOW": (208, 139, 43),
    "FAIL": (196, 64, 54),
    "MISSING": (112, 118, 128),
    "INFO": (52, 103, 170),
    "INK": (36, 40, 48),
    "MUTED": (100, 109, 122),
    "GRID": (221, 226, 232),
    "BG": (248, 250, 252),
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_num(value: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def status_rank(status: str) -> int:
    s = str(status).upper()
    if s.startswith("PASS"):
        return 3
    if s.startswith("YELLOW"):
        return 2
    if s.startswith("FAIL"):
        return 1
    return 0


def status_from_count(current: float, target: float, yellow_fraction: float = 0.75) -> str:
    if not math.isfinite(current) or current <= 0:
        return "MISSING"
    if current >= target:
        return "PASS"
    if current >= target * yellow_fraction:
        return "YELLOW"
    return "FAIL"


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\msyhbd.ttc",
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT = find_font(24)
FONT_SMALL = find_font(18)
FONT_TINY = find_font(15)
FONT_BOLD = find_font(27, bold=True)
FONT_TITLE = find_font(38, bold=True)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), str(text), font=font)
    return int(box[2] - box[0])


def wrap_label(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_px: int) -> list[str]:
    text = str(text)
    if text_width(draw, text, font) <= max_px:
        return [text]
    parts: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.replace("/", " / ").replace("_", "_ ").split()
        line = ""
        for word in words:
            trial = word if not line else f"{line} {word}"
            if text_width(draw, trial, font) <= max_px:
                line = trial
            else:
                if line:
                    parts.append(line)
                line = word
        if line:
            parts.append(line)
    if not parts:
        parts = [text[: max(1, int(max_px / 8))]]
    return parts[:4]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_px: int,
    line_gap: int = 4,
) -> int:
    x, y = xy
    lines = wrap_label(draw, text, font, max_px)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += int(font.size * 1.16) + line_gap if hasattr(font, "size") else 20
    return y


def pct(current: float, target: float) -> float:
    if not math.isfinite(current) or not math.isfinite(target) or target <= 0:
        return float("nan")
    return current / target


def add_category(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    rows.append(kwargs)


def build_fig2_categories(lesions: pd.DataFrame, counts: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add_category(
        rows,
        figure="Figure 2",
        axis="operator/readout",
        class_label="MMP soft magnetoelastic microneedle patch",
        current_n=counts.get("mmp_unique_configs", np.nan),
        target_min_n=500,
        count_basis="unique MMP config_id",
        status=status_from_count(safe_num(counts.get("mmp_unique_configs")), 500),
        note="Close to target; split/QC duplicates separately.",
    )
    add_category(
        rows,
        figure="Figure 2",
        axis="operator/readout",
        class_label="sheet piezoelectric device",
        current_n=counts.get("piezo_case_count", np.nan),
        target_min_n=150,
        count_basis="simulation benchmark readouts",
        status=status_from_count(safe_num(counts.get("piezo_case_count")), 150),
        note="Benchmark support only unless paired real data are added.",
    )
    add_category(
        rows,
        figure="Figure 2",
        axis="operator/readout",
        class_label="ultrasound strain device",
        current_n=counts.get("ultrasound_case_count", np.nan),
        target_min_n=150,
        count_basis="simulation benchmark readouts",
        status=status_from_count(safe_num(counts.get("ultrasound_case_count")), 150),
        note="Benchmark support only unless paired real data are added.",
    )

    if not lesions.empty:
        morph = lesions.copy()
        conds = [
            morph["component_count"].fillna(0) > 1,
            morph["heterogeneity"].fillna(0) >= 0.20,
            morph["contrast"].fillna(0) <= 2.0,
            morph["boundary_irregularity"].fillna(0) >= 2.0,
            morph["circularity"].fillna(0) >= 0.60,
            (morph["area_frac"].fillna(0) >= 0.65) | (morph["sigma_logE_mean_roi"].fillna(0) >= 0.35),
        ]
        labels = [
            "multifocal",
            "heterogeneous",
            "low contrast",
            "irregular boundary",
            "circular/compact",
            "blurred or broad-area",
        ]
        morph["morphology_class"] = np.select(conds, labels, default="other lesion prior")
        for label, n in morph["morphology_class"].value_counts().sort_index().items():
            add_category(
                rows,
                figure="Figure 2",
                axis="morphology phenotype",
                class_label=label,
                current_n=int(n),
                target_min_n=20,
                count_basis="exclusive heuristic class from 207 lesion maps",
                status=status_from_count(float(n), 20),
                note="Heuristic class for budget/QC; not a clinical diagnosis.",
            )
        contrast_bins = pd.cut(
            morph["contrast"].astype(float),
            bins=[-np.inf, 2.0, 8.0, np.inf],
            labels=["matched/low contrast (<=2x)", "moderate contrast (2-8x)", "high contrast (>8x)"],
        )
        for label, n in contrast_bins.value_counts().sort_index().items():
            add_category(
                rows,
                figure="Figure 2",
                axis="contrast",
                class_label=str(label),
                current_n=int(n),
                target_min_n=40,
                count_basis="lesion-to-background stiffness ratio",
                status=status_from_count(float(n), 40),
                note="Balanced training should stratify by config_id, not image instance.",
            )

    add_category(
        rows,
        figure="Figure 2",
        axis="source/support tier",
        class_label="clinical/mask-derived lesion priors",
        current_n=len(lesions) if not lesions.empty else np.nan,
        target_min_n=200,
        count_basis="lesion maps with stiffness + mask",
        status=status_from_count(float(len(lesions)) if not lesions.empty else np.nan, 200),
        note="Useful for morphology prior; not paired real MMP readout support.",
    )
    add_category(
        rows,
        figure="Figure 2",
        axis="source/support tier",
        class_label="clinical ontology labels",
        current_n=counts.get("clinical_total", np.nan),
        target_min_n=500,
        count_basis="canonical terms/labels across datasets",
        status=status_from_count(safe_num(counts.get("clinical_total")), 500),
        note="Supports prior breadth, not paired inverse-readout validation.",
    )
    add_category(
        rows,
        figure="Figure 2",
        axis="source/support tier",
        class_label="real MMP phantom/animal/ex vivo anchors",
        current_n=0,
        target_min_n=20,
        count_basis="physical paired readout + reference stiffness",
        status="MISSING",
        note="No physical MMP phantom raw/reference files were found in the inspected manifests.",
    )
    return pd.DataFrame(rows)


def build_fig3_categories(params: pd.DataFrame, depth_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not depth_metrics.empty:
        op_counts = (
            depth_metrics.groupby("operator")["n_cases"].max().sort_index()
            if "n_cases" in depth_metrics.columns
            else depth_metrics["operator"].value_counts().sort_index()
        )
        for op, n in op_counts.items():
            add_category(
                rows,
                figure="Figure 3",
                axis="operator/readout",
                class_label=str(op),
                current_n=int(n),
                target_min_n=181,
                count_basis="posterior benchmark evaluation cases per operator",
                status=status_from_count(float(n), 181),
                note="Evaluation case count; simulator prior has separate 3D config coverage.",
            )
    if not params.empty:
        for label, n in params["depth_group"].value_counts().sort_index().items():
            target = 300 if str(label) != "ood_deep" and str(label) != "OOD-deep >3 mm" else 100
            add_category(
                rows,
                figure="Figure 3",
                axis="depth",
                class_label=str(label),
                current_n=int(n),
                target_min_n=target,
                count_basis="domain_randomization/parameter_table.csv configs",
                status=status_from_count(float(n), target),
                note="Keep OOD-deep held out from ID learning curve.",
            )
        thick_bins = pd.cut(
            params["thickness_mm"].astype(float),
            bins=[-np.inf, 1.5, 3.0, np.inf],
            labels=["thin (<1.5 mm)", "medium (1.5-3.0 mm)", "thick (>3.0 mm)"],
        )
        for label, n in thick_bins.value_counts().sort_index().items():
            add_category(
                rows,
                figure="Figure 3",
                axis="thickness",
                class_label=str(label),
                current_n=int(n),
                target_min_n=300,
                count_basis="explicit z_top/z_bottom thickness truth",
                status=status_from_count(float(n), 300),
                note="Thickness truth now exists in refreshed Figure 3 parameter table.",
            )
        contrast_bins = pd.cut(
            params["contrast_x"].astype(float),
            bins=[-np.inf, 2.0, 5.0, np.inf],
            labels=["low/matched contrast (<=2x)", "moderate contrast (2-5x)", "high contrast (>5x)"],
        )
        for label, n in contrast_bins.value_counts().sort_index().items():
            add_category(
                rows,
                figure="Figure 3",
                axis="contrast",
                class_label=str(label),
                current_n=int(n),
                target_min_n=300,
                count_basis="3D stiffness contrast ratio",
                status=status_from_count(float(n), 300),
                note="If soft/hypoelastic lesions are scientifically required, add explicit contrast_x < 1 cases.",
            )
        complexity_defs = {
            "irregular boundary": params["boundary_irregularity"].astype(float) >= 0.50,
            "heterogeneous core": params["heterogeneous_core"].astype(float) >= 0.35,
            "rim stiffness": params["rim_stiffness"].astype(float) > 0,
            "multifocality": params["multifocality"].astype(float) > 0,
        }
        for label, mask in complexity_defs.items():
            n = int(mask.sum())
            add_category(
                rows,
                figure="Figure 3",
                axis="morphological complexity",
                class_label=label,
                current_n=n,
                target_min_n=250,
                count_basis="non-exclusive stressor flags from 3D parameter table",
                status=status_from_count(float(n), 250),
                note="Non-exclusive flags; one config can contribute to multiple stressors.",
            )
        contact = params["contact_pressure_kpa"].astype(float)
        noise = params["readout_noise_rel"].astype(float)
        shift = params["calibration_shift_rel"].astype(float)
        hard = (noise >= 0.06) | (shift >= 0.07) | (contact >= 17)
        baseline = (noise < 0.03) & (shift < 0.03) & (contact < 12)
        nuisance = pd.Series(np.where(hard, "hard shift", np.where(baseline, "baseline", "mild shift")))
        for label, n in nuisance.value_counts().sort_index().items():
            add_category(
                rows,
                figure="Figure 3",
                axis="contact/load/noise",
                class_label=str(label),
                current_n=int(n),
                target_min_n=250,
                count_basis="derived nuisance strata from pressure/noise/calibration shift",
                status=status_from_count(float(n), 250),
                note="Use paired seeds/sibling rows for same config_id when training.",
            )
    return pd.DataFrame(rows)


def metric_summary(fig2_metrics: pd.DataFrame, fig3_depth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not fig2_metrics.empty:
        best_mae = fig2_metrics.sort_values("MAE_kPa_mean").iloc[0]
        ours = fig2_metrics[fig2_metrics["method"].str.contains("Op-cond", case=False, na=False)]
        unet = fig2_metrics[fig2_metrics["method"].str.contains("U-Net", case=False, na=False)]
        ours_row = ours.iloc[0] if not ours.empty else best_mae
        unet_row = unet.iloc[0] if not unet.empty else None
        imp = np.nan
        if unet_row is not None:
            imp = 1 - safe_num(ours_row["MAE_kPa_mean"]) / safe_num(unet_row["MAE_kPa_mean"])
        rows.append(
            {
                "figure": "Figure 2",
                "metric_family": "2D reconstruction",
                "primary_method": str(ours_row["method"]),
                "MAE_kPa": safe_num(ours_row["MAE_kPa_mean"]),
                "SSIM": safe_num(ours_row["SSIM_mean"]),
                "Dice": safe_num(ours_row["Dice_mean"]),
                "Boundary_F1": safe_num(ours_row["Boundary_F1_mean"]),
                "relative_MAE_gain_vs_UNet": imp,
                "interpretation": "Strong point reconstruction, but data-efficiency learning curves and real MMP support are still missing.",
            }
        )
    if not fig3_depth.empty:
        ours = fig3_depth[fig3_depth["method"].eq("op_conditioned_diffusion_ours")]
        for _, row in ours.sort_values("operator").iterrows():
            rows.append(
                {
                    "figure": "Figure 3",
                    "metric_family": f"3D posterior {row['operator']}",
                    "primary_method": row["method"],
                    "MAE_zbottom_mm": safe_num(row["mae_zbottom_mm"]),
                    "P90AE_zbottom_mm": safe_num(row["p90ae_zbottom_mm"]),
                    "CRPS_zbottom_mm": safe_num(row["crps_zbottom_mm"]),
                    "Cov90": safe_num(row["cov90_zbottom"]),
                    "Width90_mm": safe_num(row["width90_zbottom_mm"]),
                    "Interval_IoU": safe_num(row["interval_iou_mean"]),
                    "AUROC_fail_1mm": safe_num(row["auroc_fail_tau_1mm"]),
                    "interpretation": "Simulation posterior benchmark is strong; final data-efficiency claim still needs learning-curve-by-N and support envelope.",
                }
            )
    return pd.DataFrame(rows)


def build_gate_table(
    counts: dict[str, Any],
    fig2_status: pd.DataFrame,
    fig2_metrics: pd.DataFrame,
    fig3_depth: pd.DataFrame,
    fig3_params: pd.DataFrame,
    locked_qc: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    phantom_missing = True
    fig2_ours = pd.DataFrame()
    if not fig2_metrics.empty:
        fig2_ours = fig2_metrics[fig2_metrics["method"].str.contains("Op-cond", case=False, na=False)]

    fig3_ours = fig3_depth[fig3_depth["method"].eq("op_conditioned_diffusion_ours")] if not fig3_depth.empty else pd.DataFrame()
    fig3_mae_pass = False
    fig3_calib_status = "MISSING"
    fig3_ood_status = "MISSING"
    if not fig3_ours.empty:
        mae_ok = (fig3_ours["mae_zbottom_mm"].astype(float) <= 0.75).all()
        iou_ok = (fig3_ours["interval_iou_mean"].astype(float) >= 0.55).all()
        fig3_mae_pass = bool(mae_ok and iou_ok)
        cov_diff = (fig3_ours["cov90_zbottom"].astype(float) - 0.90).abs()
        if (cov_diff <= 0.05).all():
            fig3_calib_status = "PASS"
        elif (cov_diff <= 0.10).all():
            fig3_calib_status = "YELLOW"
        else:
            fig3_calib_status = "FAIL"
        auroc = fig3_ours["auroc_fail_tau_1mm"].astype(float)
        if (auroc >= 0.80).all():
            fig3_ood_status = "PASS"
        elif (auroc >= 0.70).any():
            fig3_ood_status = "YELLOW"
        else:
            fig3_ood_status = "FAIL"

    visual_guardrail = bool(locked_qc.get("hybrid_realdata_sync_pass"))
    duplicate_n = int(counts.get("mmp_duplicate_hash_cases_qc", 0) or 0)
    monotonic_fail_n = int(counts.get("mmp_strain_monotonicity_fail_qc", 0) or 0)
    g0_status = "YELLOW" if duplicate_n or monotonic_fail_n else "PASS"
    rows.append(
        {
            "gate": "G0 provenance/leakage",
            "figure": "Figure 2 + Figure 3",
            "status": g0_status,
            "current_value": f"duplicate_hash={duplicate_n}; strain_monotonicity_fail={monotonic_fail_n}; figure3_visual_guardrail={visual_guardrail}",
            "ncs_threshold": "No config/latent leakage across splits; no truth-derived posterior; visual data plots from real data tables.",
            "required_action": "Quarantine duplicate-hash and monotonicity-fail cases; add config_id split audit for all learning curves.",
            "claim_level": "Simulation scaffold until split audit is attached.",
        }
    )
    rows.append(
        {
            "gate": "G1 simulator support",
            "figure": "Figure 2",
            "status": "MISSING" if phantom_missing else "PASS",
            "current_value": "real MMP phantom/animal/ex vivo anchors not found",
            "ncs_threshold": ">=90% real/phantom readouts within 95% synthetic support envelope.",
            "required_action": "Ingest >=20 physical MMP phantom anchors plus all available animal/ex vivo readouts; compute kNN/Mahalanobis/MMD support.",
            "claim_level": "Cannot claim strong sim-to-real domain randomization yet.",
        }
    )
    rows.append(
        {
            "gate": "G1 simulator support",
            "figure": "Figure 3",
            "status": "MISSING",
            "current_value": "no real MMP support-envelope table found for 3D posterior readouts",
            "ncs_threshold": ">=90% real/phantom 3D readouts within 95% synthetic readout manifold.",
            "required_action": "Create support_coverage_real_mmp.csv using readout embeddings and matched 3D simulator features.",
            "claim_level": "Simulation benchmark can remain; real-transfer claim must wait.",
        }
    )
    rows.append(
        {
            "gate": "G2 learning plateau",
            "figure": "Figure 2",
            "status": "MISSING",
            "current_value": "no fig2_metrics_by_N.csv / fitted N90 table found",
            "ncs_threshold": "Last-doubling improvement <3-5%; N90/N95/AULC reported across >=3 seeds.",
            "required_action": "Run N grid for 2D reconstruction: 50,100,200,500,1k,2k,5k,10k with config-level splits.",
            "claim_level": "Full-data benchmark only, not data-efficiency claim.",
        }
    )
    rows.append(
        {
            "gate": "G2 learning plateau",
            "figure": "Figure 3",
            "status": "MISSING",
            "current_value": "no fig3_metrics_by_N.csv / fitted N90 table found",
            "ncs_threshold": "Last-doubling improvement <3-5%; N90/N95/AULC reported across >=3 seeds.",
            "required_action": "Run N grid for 3D posterior: 100,250,500,1k,1.6k current, then 3k/5k if plateau is not reached.",
            "claim_level": "Posterior benchmark only, not data-efficiency claim.",
        }
    )
    rows.append(
        {
            "gate": "G3 method-level data efficiency",
            "figure": "Figure 2 + Figure 3",
            "status": "MISSING",
            "current_value": "DER_MAE / DER_CRPS not estimable without N90 curves",
            "ncs_threshold": "DER >=2 versus strongest baseline or clearly better CRPS/calibration/failure at same N.",
            "required_action": "Fit learning curves for U-Net, vanilla diffusion, DPS/kernel GP, and op-conditioned diffusion.",
            "claim_level": "Do not say 'twofold fewer simulations' yet.",
        }
    )
    if not fig2_ours.empty:
        row = fig2_ours.iloc[0]
        current = f"MAE={row['MAE_kPa_mean']:.3f} kPa; SSIM={row['SSIM_mean']:.3f}; Dice={row['Dice_mean']:.3f}"
        status = "YELLOW"
    else:
        current = "Figure 2 op-conditioned reconstruction metrics not found"
        status = "MISSING"
    rows.append(
        {
            "gate": "G4 Figure 2 reconstruction",
            "figure": "Figure 2",
            "status": status,
            "current_value": current,
            "ncs_threshold": "MAE/SSIM/Boundary-F1 plateau; uncertainty rejection reduces MAE/CRPS >=25%; posterior maps exported.",
            "required_action": "Export case-level final posterior mean/std maps and add uncertainty-risk curve.",
            "claim_level": "Reconstruction evidence is strong but not a complete data-efficiency package.",
        }
    )
    rows.append(
        {
            "gate": "G5 Figure 3 posterior",
            "figure": "Figure 3",
            "status": "PASS" if fig3_mae_pass else "YELLOW",
            "current_value": "; ".join(
                [
                    f"{r.operator}: MAE={r.mae_zbottom_mm:.3f}mm Cov90={r.cov90_zbottom:.3f} IoU={r.interval_iou_mean:.3f}"
                    for r in fig3_ours.itertuples()
                ]
            )
            if not fig3_ours.empty
            else "Figure 3 posterior metrics not found",
            "ncs_threshold": "In-support z_bottom MAE <=0.5-0.75mm, interval IoU >=0.55-0.60, calibrated posterior.",
            "required_action": "Keep as simulation posterior benchmark; attach learning curves and support coverage for main data-efficiency claim.",
            "claim_level": "Passes simulation benchmark; not sufficient alone for NCS data-efficiency claim.",
        }
    )
    rows.append(
        {
            "gate": "G6 calibration/sharpness",
            "figure": "Figure 3",
            "status": fig3_calib_status,
            "current_value": "; ".join(
                [
                    f"{r.operator}: Cov90={r.cov90_zbottom:.3f}, Width90={r.width90_zbottom_mm:.3f}mm"
                    for r in fig3_ours.itertuples()
                ]
            )
            if not fig3_ours.empty
            else "calibration metrics not found",
            "ncs_threshold": "|Cov90-0.90| <=0.03 ID or <=0.05 hard/OOD; width not inflated.",
            "required_action": "If any operator exceeds tolerance, calibrate intervals on validation split and report pre/post calibration.",
            "claim_level": "Use exact operator-specific language.",
        }
    )
    rows.append(
        {
            "gate": "G7 OOD/failure utility",
            "figure": "Figure 3",
            "status": fig3_ood_status,
            "current_value": "; ".join(
                [
                    f"{r.operator}: AUROC_fail_1mm={r.auroc_fail_tau_1mm:.3f}"
                    for r in fig3_ours.itertuples()
                ]
            )
            if not fig3_ours.empty
            else "failure metrics not found",
            "ncs_threshold": "AUROC_fail >=0.75-0.80; risk at retained 80% improves; AUSE improves >=15-20%.",
            "required_action": "Add failure prevalence/AUPRC and retained-risk curves by depth/OOD group.",
            "claim_level": "May support failure-screening claim if prevalence diagnostics are attached.",
        }
    )
    rows.append(
        {
            "gate": "G8 projection consistency",
            "figure": "Figure 2 + Figure 3",
            "status": "MISSING",
            "current_value": "projection arrays exist, but no projection_consistency_metrics.csv found",
            "ncs_threshold": "NMAE(Eapp2D, Pz[E3D]) <=0.10-0.15; SSIM >=0.80-0.85; centroid shift <=0.5-1.0mm.",
            "required_action": "Compute paired 2D/3D projection QC before merging Figure 2 and Figure 3 story.",
            "claim_level": "Do not merge as a single posterior framework until this gate is measured.",
        }
    )
    return pd.DataFrame(rows)


def build_improvement_queue() -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "figure": "Figure 2 + Figure 3",
            "gap": "No actual learning-curve-by-N files",
            "action": "Run stratified N-grid training/evaluation with >=3 seeds and config_id-level splits.",
            "target_metric": "N90, N95, AULC, last-doubling gain, seed SD",
            "target_threshold": "last-doubling gain <3-5%; DER >=2 if claimed",
            "output_file": "fig2_metrics_by_N.csv; fig3_metrics_by_N.csv; learning_curve_fits.csv",
            "automation_level": "requires training jobs",
        },
        {
            "priority": 2,
            "figure": "Figure 2 + Figure 3",
            "gap": "Real MMP support envelope missing",
            "action": "Ingest physical phantom/animal/ex vivo readouts and compute readout-feature support coverage against synthetic manifold.",
            "target_metric": "support_score, kNN distance, Mahalanobis distance, MMD, conformal p-value",
            "target_threshold": ">=90% real/phantom readouts inside 95% synthetic support envelope",
            "output_file": "support_coverage_real_mmp.csv",
            "automation_level": "requires raw real data",
        },
        {
            "priority": 3,
            "figure": "Figure 2",
            "gap": "Publication-ready posterior maps missing",
            "action": "Export case-level posterior mean/std maps and uncertainty rejection curves for selected lesion cases.",
            "target_metric": "MAE, SSIM, Dice, Boundary-F1, uncertainty-retained risk",
            "target_threshold": "uncertainty rejection reduces MAE/CRPS >=25%",
            "output_file": "fig2_case_level_posteriors.csv; fig2_uncertainty_risk.csv",
            "automation_level": "requires inference/export",
        },
        {
            "priority": 4,
            "figure": "Figure 3",
            "gap": "Projection consistency before Figure 2/3 merge missing",
            "action": "Project 3D posterior mean to 2D apparent stiffness and compare with Figure 2 Eapp maps.",
            "target_metric": "projection NMAE, SSIM, centroid shift, area difference",
            "target_threshold": "NMAE <=0.10-0.15; SSIM >=0.80-0.85; centroid shift <=1mm",
            "output_file": "projection_consistency_metrics.csv",
            "automation_level": "can be scripted after paired cases are defined",
        },
        {
            "priority": 5,
            "figure": "Figure 3",
            "gap": "Calibration tolerance should be operator-specific",
            "action": "Report pre/post conformal or validation calibration without test leakage.",
            "target_metric": "Cov90, Width90, UCE, ENCE, PIT shape",
            "target_threshold": "|Cov90-0.90| <=0.03 ID or <=0.05 hard/OOD",
            "output_file": "calibration_prepost_metrics.csv",
            "automation_level": "can be scripted from validation posterior samples",
        },
        {
            "priority": 6,
            "figure": "Figure 2 + Figure 3",
            "gap": "Duplicate-hash and monotonicity-fail MMP cases in source QC",
            "action": "Quarantine failed cases and enforce split by latent config_id before any data-efficiency training.",
            "target_metric": "duplicate_hash_cases=0 in train/test overlap; monotonicity_fail excluded or labeled",
            "target_threshold": "zero leakage; all exclusions logged",
            "output_file": "split_leakage_audit.csv; qc_exclusion_manifest.csv",
            "automation_level": "can be scripted",
        },
    ]
    return pd.DataFrame(rows)


def build_learning_curve_plan() -> pd.DataFrame:
    rows = []
    configs = [
        ("Figure 2", "2D Eapp reconstruction", [50, 100, 200, 500, 1000, 2000, 5000, 10000], ["Baseline U-Net", "Vanilla diffusion", "Op-cond. diffusion + mask"]),
        ("Figure 3", "3D z_bottom posterior", [100, 250, 500, 1000, 1600, 3000, 5000], ["direct_scalar_regression", "unet_3d", "kernel_gp", "vanilla_3d_diffusion", "dps_inverse_diffusion", "op_conditioned_diffusion_ours"]),
    ]
    for figure, task, ns, methods in configs:
        for method in methods:
            for n in ns:
                for seed in [20260604, 20260605, 20260606]:
                    rows.append(
                        {
                            "figure": figure,
                            "task": task,
                            "method": method,
                            "train_N": n,
                            "seed": seed,
                            "split_rule": "group by latent config_id; OOD-deep held out for OOD gate",
                            "primary_metrics": "MAE, CRPS/WIS90, Cov90/Width90, AUROC_fail, retained-risk",
                        }
                    )
    return pd.DataFrame(rows)


def build_api_payloads(out: Path, visual_paths: dict[str, str], gate_table: pd.DataFrame, class_summary: pd.DataFrame) -> None:
    api_dir = out / "image_api_payloads"
    api_dir.mkdir(parents=True, exist_ok=True)
    compact_gates = gate_table[["gate", "figure", "status", "current_value", "claim_level"]].to_dict(orient="records")
    compact_classes = class_summary[["figure", "axis", "n_classes", "min_current_n", "overall_status"]].to_dict(orient="records")
    base_prompt = {
        "visual_goal": "Create a Nature Computational Science-style data-efficiency readiness visualization for Figure 2 and Figure 3. Do not invent values. Use only the provided gate table and class counts. Make missing gates visually explicit.",
        "data_constraints": {
            "gate_table": compact_gates,
            "class_summary": compact_classes,
            "local_reference_images": visual_paths,
        },
        "style": "clean scientific dashboard, white background, restrained color, panel labels, exact numeric callouts, no decorative stock imagery",
    }
    openai_payload = {
        "not_executed_reason": "No OPENAI_API_KEY was present in the local environment during audit.",
        "requested_user_label": "Image 2.0",
        "official_openai_image_model_at_audit_time": "gpt-image-1.5 or chatgpt-image-latest; replace only if your account exposes a newer Image 2.0 model id.",
        "responses_api_payload": {
            "model": "gpt-5.5",
            "input": base_prompt,
            "tools": [{"type": "image_generation"}],
        },
        "images_api_payload": {
            "model": "gpt-image-1.5",
            "prompt": json.dumps(base_prompt, ensure_ascii=False),
            "size": "1536x1024",
            "quality": "high",
            "n": 1,
        },
    }
    nano_payload = {
        "not_executed_reason": "No NanoBanana connector/tool or API key was available in this Codex session.",
        "requested_user_label": "NanoBanana API",
        "model": "NANOBANANA_MODEL_ID_FROM_PROVIDER",
        "input": base_prompt,
        "negative_instruction": "Do not synthesize new data, do not change PASS/YELLOW/FAIL/MISSING labels, and do not replace numeric tables with decorative graphics.",
    }
    (api_dir / "openai_image_payload.json").write_text(json.dumps(openai_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (api_dir / "nanobanana_image_payload.json").write_text(json.dumps(nano_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    work = work.fillna("")
    cols = [str(c) for c in work.columns]

    def clean(value: Any) -> str:
        text = str(value).replace("\n", " ").replace("|", "/")
        return text

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(clean(row[c]) for c in work.columns) + " |")
    return "\n".join(lines)


def draw_status_badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], status: str, w: int = 122, h: int = 32) -> None:
    s = str(status).upper()
    if s.startswith("PASS"):
        color = COLORS["PASS"]
        label = "PASS"
    elif s.startswith("YELLOW"):
        color = COLORS["YELLOW"]
        label = "YELLOW"
    elif s.startswith("FAIL"):
        color = COLORS["FAIL"]
        label = "FAIL"
    else:
        color = COLORS["MISSING"]
        label = "MISSING"
    x, y = xy
    draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=color)
    tw = text_width(draw, label, FONT_SMALL)
    draw.text((x + (w - tw) // 2, y + 6), label, font=FONT_SMALL, fill=(255, 255, 255))


def make_scorecard(gates: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1800, 1380), COLORS["BG"])
    draw = ImageDraw.Draw(img)
    draw.text((60, 42), "NCS Data-Efficiency Readiness Scorecard", font=FONT_TITLE, fill=COLORS["INK"])
    draw.text(
        (60, 92),
        "Figure 2/3 audit: pass strong claims only when learning curves, support coverage, calibration, and OOD utility close.",
        font=FONT_SMALL,
        fill=COLORS["MUTED"],
    )
    x0, y0 = 60, 150
    col = [0, 360, 575, 725, 1190]
    headers = ["Gate", "Figure", "Status", "Current evidence", "Action before strong NCS claim"]
    for i, header in enumerate(headers):
        draw.text((x0 + col[i], y0), header, font=FONT_BOLD, fill=COLORS["INK"])
    y = y0 + 48
    row_h = 88
    for idx, row in gates.iterrows():
        if y + row_h > 1300:
            break
        fill = (255, 255, 255) if idx % 2 == 0 else (244, 247, 251)
        draw.rounded_rectangle([x0, y - 8, 1740, y + row_h - 12], radius=7, fill=fill, outline=COLORS["GRID"])
        draw_wrapped(draw, (x0 + col[0], y), row["gate"], FONT_SMALL, COLORS["INK"], 330)
        draw_wrapped(draw, (x0 + col[1], y), row["figure"], FONT_TINY, COLORS["MUTED"], 190)
        draw_status_badge(draw, (x0 + col[2], y + 3), row["status"])
        draw_wrapped(draw, (x0 + col[3], y), row["current_value"], FONT_TINY, COLORS["INK"], 440)
        draw_wrapped(draw, (x0 + col[4], y), row["required_action"], FONT_TINY, COLORS["INK"], 520)
        y += row_h
    draw.text((60, 1325), f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from local CSV/JSON only.", font=FONT_TINY, fill=COLORS["MUTED"])
    img.save(path)


def make_category_coverage(categories: pd.DataFrame, path: Path) -> None:
    rows = categories.copy()
    rows = rows.sort_values(["figure", "axis", "class_label"])
    img_h = max(1250, 185 + len(rows) * 66 + 80)
    img = Image.new("RGB", (1900, img_h), COLORS["BG"])
    draw = ImageDraw.Draw(img)
    draw.text((60, 42), "Domain-Randomization Class Coverage", font=FONT_TITLE, fill=COLORS["INK"])
    draw.text(
        (60, 92),
        "Counts are per axis; stressor classes can be non-exclusive. Targets are minimum QC budgets, not proof of learning-curve saturation.",
        font=FONT_SMALL,
        fill=COLORS["MUTED"],
    )
    left = 80
    top = 155
    bar_x = 650
    bar_w = 760
    label_w = 560
    y = top
    for _, row in rows.iterrows():
        if y > img_h - 80:
            break
        cur = safe_num(row["current_n"], 0)
        target = safe_num(row["target_min_n"], 1)
        frac = min(cur / target if target else 0, 1.35)
        st = str(row["status"]).upper()
        color = COLORS["PASS"] if st.startswith("PASS") else COLORS["YELLOW"] if st.startswith("YELLOW") else COLORS["FAIL"] if st.startswith("FAIL") else COLORS["MISSING"]
        draw.text((left, y), f"{row['figure']} | {row['axis']}", font=FONT_TINY, fill=COLORS["MUTED"])
        draw_wrapped(draw, (left, y + 20), str(row["class_label"]), FONT_SMALL, COLORS["INK"], label_w)
        draw.rounded_rectangle([bar_x, y + 12, bar_x + bar_w, y + 42], radius=5, fill=(232, 237, 243))
        draw.rounded_rectangle([bar_x, y + 12, bar_x + int(bar_w * min(frac, 1.0)), y + 42], radius=5, fill=color)
        draw.text((bar_x + bar_w + 30, y + 13), f"n={int(cur) if math.isfinite(cur) else 'NA'} / target {int(target)}", font=FONT_SMALL, fill=COLORS["INK"])
        draw_status_badge(draw, (bar_x + bar_w + 315, y + 8), st, w=110, h=30)
        y += 66
    img.save(path)


def make_metric_dashboard(metrics: pd.DataFrame, gates: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1800, 1300), COLORS["BG"])
    draw = ImageDraw.Draw(img)
    draw.text((60, 42), "Figure 2 / Figure 3 Data-Efficiency Evidence Dashboard", font=FONT_TITLE, fill=COLORS["INK"])
    draw.text((60, 92), "Exact local metrics are shown separately from missing data-efficiency gates.", font=FONT_SMALL, fill=COLORS["MUTED"])

    panel_y = 150
    draw.rounded_rectangle([60, panel_y, 870, 560], radius=8, fill=(255, 255, 255), outline=COLORS["GRID"])
    draw.text((90, panel_y + 24), "Figure 2: 2D apparent rigidity reconstruction", font=FONT_BOLD, fill=COLORS["INK"])
    fig2 = metrics[metrics["figure"].eq("Figure 2")]
    if not fig2.empty:
        r = fig2.iloc[0]
        vals = [
            ("MAE kPa", safe_num(r.get("MAE_kPa")), 7.0, False),
            ("SSIM", safe_num(r.get("SSIM")), 1.0, True),
            ("Dice", safe_num(r.get("Dice")), 1.0, True),
            ("Boundary-F1", safe_num(r.get("Boundary_F1")), 1.0, True),
        ]
        y = panel_y + 88
        for label, val, maxv, high_good in vals:
            draw.text((90, y), label, font=FONT_SMALL, fill=COLORS["INK"])
            x = 270
            w = 430
            frac = max(0.0, min(val / maxv, 1.0)) if math.isfinite(val) else 0.0
            color = COLORS["PASS"] if high_good else COLORS["YELLOW"]
            draw.rounded_rectangle([x, y + 2, x + w, y + 28], radius=5, fill=(232, 237, 243))
            draw.rounded_rectangle([x, y + 2, x + int(w * frac), y + 28], radius=5, fill=color)
            draw.text((x + w + 24, y), f"{val:.3f}" if math.isfinite(val) else "NA", font=FONT_SMALL, fill=COLORS["INK"])
            y += 58
        draw_wrapped(draw, (90, panel_y + 332), str(r["interpretation"]), FONT_SMALL, COLORS["MUTED"], 720)

    draw.rounded_rectangle([930, panel_y, 1740, 560], radius=8, fill=(255, 255, 255), outline=COLORS["GRID"])
    draw.text((960, panel_y + 24), "Figure 3: 3D posterior benchmark (ours)", font=FONT_BOLD, fill=COLORS["INK"])
    fig3 = metrics[metrics["figure"].eq("Figure 3")]
    y = panel_y + 86
    for _, r in fig3.iterrows():
        op = str(r["metric_family"]).replace("3D posterior ", "")
        mae = safe_num(r.get("MAE_zbottom_mm"))
        cov = safe_num(r.get("Cov90"))
        iou = safe_num(r.get("Interval_IoU"))
        draw.text((960, y), op, font=FONT_SMALL, fill=COLORS["INK"])
        draw.text((1105, y), f"MAE {mae:.3f} mm", font=FONT_SMALL, fill=COLORS["PASS"] if mae <= 0.75 else COLORS["YELLOW"])
        draw.text((1325, y), f"Cov90 {cov:.3f}", font=FONT_SMALL, fill=COLORS["PASS"] if abs(cov - 0.90) <= 0.05 else COLORS["YELLOW"])
        draw.text((1530, y), f"IoU {iou:.3f}", font=FONT_SMALL, fill=COLORS["PASS"] if iou >= 0.55 else COLORS["YELLOW"])
        y += 60
    draw_wrapped(draw, (960, panel_y + 332), "These values support a simulation posterior benchmark, but do not replace the missing N-grid and real-support gates.", FONT_SMALL, COLORS["MUTED"], 700)

    panel2_y = 625
    draw.rounded_rectangle([60, panel2_y, 1740, 1170], radius=8, fill=(255, 255, 255), outline=COLORS["GRID"])
    draw.text((90, panel2_y + 24), "Missing gates that block a strong Nature Computational Science data-efficiency claim", font=FONT_BOLD, fill=COLORS["INK"])
    missing = gates[gates["status"].isin(["MISSING", "FAIL", "YELLOW"])].head(8)
    y = panel2_y + 82
    for _, row in missing.iterrows():
        draw_status_badge(draw, (90, y), row["status"], w=110, h=30)
        draw_wrapped(draw, (230, y), f"{row['gate']} ({row['figure']}): {row['required_action']}", FONT_SMALL, COLORS["INK"], 1450)
        y += 52
    draw.text((60, 1225), "Local rendering; no generative image model was called because API keys/connectors were not available.", font=FONT_TINY, fill=COLORS["MUTED"])
    img.save(path)


def class_summary(categories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (figure, axis), g in categories.groupby(["figure", "axis"], sort=True):
        statuses = list(g["status"].astype(str))
        if all(s.startswith("PASS") for s in statuses):
            overall = "PASS"
        elif any(s.startswith("MISSING") for s in statuses):
            overall = "MISSING"
        elif any(s.startswith("FAIL") for s in statuses):
            overall = "FAIL"
        else:
            overall = "YELLOW"
        rows.append(
            {
                "figure": figure,
                "axis": axis,
                "n_classes": int(len(g)),
                "min_current_n": safe_num(g["current_n"].replace("not found", np.nan).astype(float).min()),
                "max_current_n": safe_num(g["current_n"].replace("not found", np.nan).astype(float).max()),
                "overall_status": overall,
                "class_labels": "; ".join(g["class_label"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    gates: pd.DataFrame,
    categories: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    queue: pd.DataFrame,
    visual_paths: dict[str, str],
) -> None:
    overall_pass = bool((gates["status"] == "PASS").all())
    fig2_axes = summary[summary["figure"].eq("Figure 2")]
    fig3_axes = summary[summary["figure"].eq("Figure 3")]
    fig2_n_classes = int(fig2_axes["n_classes"].sum()) if not fig2_axes.empty else 0
    fig3_n_classes = int(fig3_axes["n_classes"].sum()) if not fig3_axes.empty else 0
    lines: list[str] = []
    lines.append("# NCS data-efficiency audit for Figure 2 and Figure 3\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("## Executive decision\n")
    if overall_pass:
        lines.append("- Current evidence passes all pre-specified gates for a strong NCS data-efficiency claim.\n")
    else:
        lines.append("- Current evidence does **not yet** pass all gates for a strong Nature Computational Science data-efficiency/sim-to-real claim.\n")
        lines.append("- It is suitable for a simulation-grounded Figure 2/3 benchmark, with missing gates explicitly labeled.\n")
    lines.append(f"- Figure 2 currently uses {len(fig2_axes)} domain-randomization axes and {fig2_n_classes} axis-level classes.\n")
    lines.append(f"- Figure 3 currently uses {len(fig3_axes)} domain-randomization axes and {fig3_n_classes} axis-level classes.\n")
    lines.append("- Class counts are per axis; some Figure 3 stressor classes are intentionally non-exclusive.\n")
    lines.append("\n## Gate table\n")
    lines.append(markdown_table(gates))
    lines.append("\n\n## Class summary\n")
    lines.append(markdown_table(summary))
    lines.append("\n\n## Primary metric snapshot\n")
    lines.append(markdown_table(metrics))
    lines.append("\n\n## Immediate improvement queue\n")
    lines.append(markdown_table(queue))
    lines.append("\n\n## Generated visualizations\n")
    for name, path in visual_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("\n\n## Interpretation for manuscript claims\n")
    lines.append("- Strong wording allowed now: Figure 3 has strong simulation posterior benchmark metrics for the op-conditioned diffusion model; Figure 2 has strong point reconstruction evidence on the available benchmark table.")
    lines.append("- Strong wording not allowed yet: 'domain randomization achieved sim-to-real transfer' or 'twofold data-efficient' until learning curves and real support coverage pass.")
    lines.append("- Required NCS-facing metrics: N90/N95/AULC, data-efficiency ratio, support coverage, Cov90/Width90/UCE/ENCE, AUROC/AUPRC failure utility, and 2D/3D projection consistency.")
    (out / "NCS_DATA_EFFICIENCY_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "visualizations").mkdir(parents=True, exist_ok=True)

    counts = read_json(PREV_OUT / "fig23_counts_from_reference_data.json", default={}) or {}
    fig2_status = read_csv(FIG2 / "tables" / "panel_data_status.csv")
    fig2_lesions = read_csv(FIG2 / "tables" / "lesion_case_inventory_metrics.csv")
    fig2_metrics = read_csv(FIG2 / "tables" / "computed_panel_e_metrics_summary.csv")
    fig3_params = read_csv(FIG3_DATA / "domain_randomization" / "parameter_table.csv")
    fig3_depth = read_csv(FIG3_DATA / "metrics" / "depth_metrics.csv")
    locked_qc = read_json(FIG3_FIG / "locked_overlay_qc.json", default={}) or {}

    fig2_categories = build_fig2_categories(fig2_lesions, counts)
    fig3_categories = build_fig3_categories(fig3_params, fig3_depth)
    categories = pd.concat([fig2_categories, fig3_categories], ignore_index=True)
    categories["coverage_fraction"] = categories.apply(
        lambda r: pct(safe_num(r["current_n"]), safe_num(r["target_min_n"])), axis=1
    )
    summary = class_summary(categories)
    metrics = metric_summary(fig2_metrics, fig3_depth)
    gates = build_gate_table(counts, fig2_status, fig2_metrics, fig3_depth, fig3_params, locked_qc)
    queue = build_improvement_queue()
    learning_plan = build_learning_curve_plan()

    categories.to_csv(OUT / "ncs_domain_randomization_class_counts.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "ncs_category_axis_summary.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(OUT / "ncs_gate_decisions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "ncs_primary_metric_snapshot.csv", index=False, encoding="utf-8-sig")
    queue.to_csv(OUT / "ncs_improvement_queue.csv", index=False, encoding="utf-8-sig")
    learning_plan.to_csv(OUT / "ncs_learning_curve_job_plan.csv", index=False, encoding="utf-8-sig")
    (OUT / "ncs_audit_summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "overall_strong_ncs_ready": bool((gates["status"] == "PASS").all()),
                "figure2_axes": int((summary["figure"] == "Figure 2").sum()),
                "figure2_axis_level_classes": int(summary.loc[summary["figure"] == "Figure 2", "n_classes"].sum()),
                "figure3_axes": int((summary["figure"] == "Figure 3").sum()),
                "figure3_axis_level_classes": int(summary.loc[summary["figure"] == "Figure 3", "n_classes"].sum()),
                "gate_status_counts": gates["status"].value_counts().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    visuals = {
        "scorecard": str(OUT / "visualizations" / "ncs_readiness_scorecard.png"),
        "coverage": str(OUT / "visualizations" / "domain_randomization_class_coverage.png"),
        "dashboard": str(OUT / "visualizations" / "figure2_figure3_data_efficiency_dashboard.png"),
    }
    make_scorecard(gates, Path(visuals["scorecard"]))
    make_category_coverage(categories, Path(visuals["coverage"]))
    make_metric_dashboard(metrics, gates, Path(visuals["dashboard"]))
    build_api_payloads(OUT, visuals, gates, summary)
    write_report(OUT, gates, categories, summary, metrics, queue, visuals)

    print(json.dumps({
        "out": str(OUT),
        "gates": gates["status"].value_counts().to_dict(),
        "figure2_classes": int(summary.loc[summary["figure"] == "Figure 2", "n_classes"].sum()),
        "figure3_classes": int(summary.loc[summary["figure"] == "Figure 3", "n_classes"].sum()),
        "visuals": visuals,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
