from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"H:\My Drive")
OUT = ROOT / "Manuscript" / "Sparse reconstruction" / "Data" / "Data efficiency" / "fig23_data_efficiency_outputs"


SOURCES = {
    "pdf_extract": ROOT
    / "Manuscript"
    / "Sparse reconstruction"
    / "Data"
    / "Data efficiency"
    / "pdf_pages_25_68_extracted_text.txt",
    "fig3_manifest": ROOT / "MN_simulation" / "data audit 050726 (1+2)" / "fig3_synthetic_manifest.csv",
    "fig3_qc_summary": ROOT / "MN_simulation" / "data audit 050726 (1+2)" / "fig3_qc_summary.json",
    "comsol_summary": ROOT / "MN_simulation" / "data audit 050726 (1+2)" / "comsol_data_summary_by_group.csv",
    "rigidity_manifest": ROOT / "3D_rigidity_ mapping" / "derived" / "manifests" / "manifest_cases.csv",
    "rigidity_scalar": ROOT / "3D_rigidity_ mapping" / "derived" / "labels" / "manifest_cases_with_scalar.csv",
    "piezo_manifest": ROOT / "MN_simulation" / "piezo_passive_benchmark" / "outputs" / "n7_matched_20260525" / "piezo_case_manifest.csv",
    "piezo_full_manifest": ROOT
    / "MN_simulation"
    / "piezo_passive_benchmark"
    / "outputs"
    / "n7_matched_20260525"
    / "full_comsol"
    / "piezo_full_comsol_all_manifest.csv",
    "piezo_summary": ROOT / "MN_simulation" / "piezo_passive_benchmark" / "outputs" / "n7_matched_20260525" / "piezo_case_manifest_summary.json",
    "ultrasound_manifest": ROOT / "MN_simulation" / "ultrasound_strain_benchmark" / "outputs" / "n7_matched_20260525" / "us_case_manifest.csv",
    "ultrasound_full_manifest": ROOT
    / "MN_simulation"
    / "ultrasound_strain_benchmark"
    / "outputs"
    / "n7_matched_20260525"
    / "full_comsol"
    / "us_full_comsol_all_manifest.csv",
    "ultrasound_summary": ROOT / "MN_simulation" / "ultrasound_strain_benchmark" / "outputs" / "n7_matched_20260525" / "us_case_manifest_summary.json",
    "clinical_label_summary": ROOT / "Clincal_ontology" / "harmonized_annotations_label_summary.csv",
    "clinical_split_summary": ROOT / "Clincal_ontology" / "final_split_label_summary.csv",
    "figure2_public_support": ROOT / "Clincal_ontology" / "derived" / "figure2" / "figure2_v11_data_summary.json",
    "rigidity_srcnn_summary": ROOT / "rigidity_interpolation" / "analysis_reports" / "srcnn_8x8_reconstruction_report" / "validation_summary.json",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_unique(df: pd.DataFrame, col: str) -> int:
    return int(df[col].nunique(dropna=True)) if col in df.columns and not df.empty else 0


def safe_values(df: pd.DataFrame, col: str) -> list[Any]:
    if col not in df.columns or df.empty:
        return []
    vals = df[col].dropna().unique().tolist()
    return sorted(vals, key=lambda x: str(x))


def count_by(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if df.empty or not cols:
        return pd.DataFrame(columns=[*cols, "n_rows"])
    return df.groupby(cols, dropna=False).size().reset_index(name="n_rows").sort_values(cols)


def bucket_depth(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return "unknown"
    if x <= 0.75:
        return "shallow_0_0p75mm"
    if x <= 1.5:
        return "mid_0p75_1p5mm"
    if x <= 3.0:
        return "deep_1p5_3mm"
    return "ood_deep_gt3mm"


def bucket_contrast(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return "unknown"
    if x < -0.2:
        return "soft_or_hypoelastic"
    if x <= 0.2:
        return "matched_low_contrast"
    return "stiff_or_hyperelastic"


def bucket_float(v: Any, bins: list[tuple[float, float, str]], default: str = "unknown") -> str:
    try:
        x = float(v)
    except Exception:
        return default
    for lo, hi, name in bins:
        if lo <= x <= hi:
            return name
    return default


def source_status(path: Path) -> str:
    return "FOUND" if path.exists() else "MISSING"


def add_inventory_row(rows: list[dict[str, Any]], source_key: str, role: str, details: dict[str, Any] | None = None) -> None:
    path = SOURCES[source_key]
    row = {
        "source_key": source_key,
        "role": role,
        "status": source_status(path),
        "path": str(path),
    }
    if details:
        row.update(details)
    rows.append(row)


def summarize_reference_data() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fig3 = read_csv(SOURCES["fig3_manifest"])
    fig3_qc = read_json(SOURCES["fig3_qc_summary"])
    rigidity = read_csv(SOURCES["rigidity_manifest"])
    rigidity_scalar = read_csv(SOURCES["rigidity_scalar"])
    piezo = read_csv(SOURCES["piezo_manifest"])
    piezo_full = read_csv(SOURCES["piezo_full_manifest"])
    piezo_summary = read_json(SOURCES["piezo_summary"])
    us = read_csv(SOURCES["ultrasound_manifest"])
    us_full = read_csv(SOURCES["ultrasound_full_manifest"])
    us_summary = read_json(SOURCES["ultrasound_summary"])
    clinical = read_csv(SOURCES["clinical_label_summary"])
    clinical_split = read_csv(SOURCES["clinical_split_summary"])
    fig2_support = read_json(SOURCES["figure2_public_support"])
    srcnn_summary = read_json(SOURCES["rigidity_srcnn_summary"])

    inventory_rows: list[dict[str, Any]] = []
    add_inventory_row(
        inventory_rows,
        "fig3_manifest",
        "MMP / MN simulation manifest for Figure 2-3",
        {
            "rows": len(fig3),
            "unique_case_source_group": safe_unique(fig3, "case_source_group"),
            "unique_case_id": safe_unique(fig3, "case_id"),
            "method_types": ";".join(map(str, safe_values(fig3, "method_type"))),
        },
    )
    add_inventory_row(
        inventory_rows,
        "fig3_qc_summary",
        "Existing Figure 3 QC summary",
        {
            "n_method1_cases": fig3_qc.get("n_method1_cases"),
            "n_method2_cases": fig3_qc.get("n_method2_cases"),
            "n_trainable_cases": fig3_qc.get("n_trainable_cases"),
            "n_duplicate_hash_cases": fig3_qc.get("n_duplicate_hash_cases"),
            "n_strain_monotonicity_fail": fig3_qc.get("n_strain_monotonicity_fail"),
        },
    )
    add_inventory_row(
        inventory_rows,
        "rigidity_manifest",
        "3D rigidity mapping base grid",
        {
            "rows": len(rigidity),
            "depth_levels": safe_unique(rigidity, "depth_mm"),
            "strain_levels": safe_unique(rigidity, "strain_float"),
            "E_levels": safe_unique(rigidity, "E_tag"),
        },
    )
    add_inventory_row(
        inventory_rows,
        "rigidity_scalar",
        "3D rigidity mapping scalar labels",
        {
            "rows": len(rigidity_scalar),
            "depth_levels": safe_unique(rigidity_scalar, "depth_mm"),
            "strain_levels": safe_unique(rigidity_scalar, "strain_float"),
            "E_levels": safe_unique(rigidity_scalar, "E_tag"),
        },
    )
    add_inventory_row(
        inventory_rows,
        "piezo_manifest",
        "Sheet piezoelectric simulated benchmark manifest",
        {
            "rows": len(piezo),
            "case_count_from_summary": piezo_summary.get("case_count"),
            "condition_count": piezo_summary.get("condition_count"),
            "sample_count": piezo_summary.get("sample_count"),
            "contact_modes": ";".join(map(str, piezo_summary.get("contact_modes", []))),
            "strain_levels": len(piezo_summary.get("strains", [])),
        },
    )
    add_inventory_row(
        inventory_rows,
        "piezo_full_manifest",
        "Sheet piezoelectric full COMSOL export manifest",
        {"rows": len(piezo_full), "unique_cases": safe_unique(piezo_full, "case_id")},
    )
    add_inventory_row(
        inventory_rows,
        "ultrasound_manifest",
        "Ultrasound strain simulated benchmark manifest",
        {
            "rows": len(us),
            "case_count_from_summary": us_summary.get("case_count"),
            "condition_count": us_summary.get("condition_count"),
            "sample_count": us_summary.get("sample_count"),
            "contact_modes": ";".join(map(str, us_summary.get("contact_modes", []))),
            "compression_levels": len(us_summary.get("compressions", [])),
        },
    )
    add_inventory_row(
        inventory_rows,
        "ultrasound_full_manifest",
        "Ultrasound strain full COMSOL export manifest",
        {"rows": len(us_full), "unique_cases": safe_unique(us_full, "case_id")},
    )
    add_inventory_row(
        inventory_rows,
        "clinical_label_summary",
        "Clinical ontology label support",
        {
            "rows": len(clinical),
            "datasets": safe_unique(clinical, "dataset"),
            "canonical_terms": safe_unique(clinical, "canonical_term"),
            "total_images_or_lesions": int(clinical["n"].sum()) if "n" in clinical.columns and not clinical.empty else None,
        },
    )
    add_inventory_row(
        inventory_rows,
        "clinical_split_summary",
        "Clinical ontology split label support",
        {
            "rows": len(clinical_split),
            "splits": safe_unique(clinical_split, "split"),
            "canonical_terms": safe_unique(clinical_split, "canonical_term"),
        },
    )
    add_inventory_row(
        inventory_rows,
        "figure2_public_support",
        "Figure 2 public embedding support summary",
        {
            "projection_n": fig2_support.get("panel_f", {}).get("projection_n"),
            "target_n": fig2_support.get("panel_f", {}).get("target_n"),
            "target_public_overlap_fraction": fig2_support.get("panel_f", {}).get("target_public_overlap_fraction"),
            "effective_sample_size_fraction": fig2_support.get("panel_f", {}).get("effective_sample_size_fraction"),
        },
    )
    add_inventory_row(
        inventory_rows,
        "rigidity_srcnn_summary",
        "Existing 2D sparse reconstruction pilot",
        {
            "test_sample_count": srcnn_summary.get("test_sample_count"),
            "subtype_count": srcnn_summary.get("subtype_count"),
            "pixel_accuracy_available": srcnn_summary.get("pixel_accuracy_available"),
        },
    )

    inventory_df = pd.DataFrame(inventory_rows)

    category_rows: list[dict[str, Any]] = []

    if not fig3.empty:
        method_counts = count_by(fig3, "method_type", "target_type")
        for _, r in method_counts.iterrows():
            category_rows.append(
                {
                    "source": "fig3_synthetic_manifest",
                    "category_axis": "MMP_simulation_method_target",
                    "category": f"{r.get('method_type')} | {r.get('target_type')}",
                    "n_rows": int(r["n_rows"]),
                    "n_unique_configs": safe_unique(
                        fig3[(fig3.get("method_type") == r.get("method_type")) & (fig3.get("target_type") == r.get("target_type"))],
                        "case_source_group",
                    ),
                }
            )
        for col in ["disease_or_family", "condition_state", "geometry_id", "gap_dir_name", "strain", "split_role_prior"]:
            for _, r in count_by(fig3, col).iterrows():
                category_rows.append(
                    {
                        "source": "fig3_synthetic_manifest",
                        "category_axis": col,
                        "category": r[col],
                        "n_rows": int(r["n_rows"]),
                        "n_unique_configs": safe_unique(fig3[fig3[col] == r[col]], "case_source_group"),
                    }
                )

    if not rigidity.empty:
        tmp = rigidity.copy()
        tmp["depth_group"] = tmp["depth_mm"].apply(bucket_depth) if "depth_mm" in tmp.columns else "unknown"
        for col in ["depth_group", "depth_mm", "E_tag", "strain_float"]:
            for _, r in count_by(tmp, col).iterrows():
                category_rows.append(
                    {
                        "source": "3D_rigidity_mapping",
                        "category_axis": col,
                        "category": r[col],
                        "n_rows": int(r["n_rows"]),
                        "n_unique_configs": safe_unique(tmp[tmp[col] == r[col]], "case_id"),
                    }
                )

    for label, df, full_df in [
        ("PIEZO_PASSIVE", piezo, piezo_full),
        ("ULTRASOUND_STRAIN", us, us_full),
    ]:
        if df.empty:
            continue
        for col in ["family_id", "condition_type", "A_contrast", "H_heterogeneity", "contact_mode", "u_load"]:
            if col in df.columns:
                for _, r in count_by(df, col).iterrows():
                    category_rows.append(
                        {
                            "source": label,
                            "category_axis": col,
                            "category": r[col],
                            "n_rows": int(r["n_rows"]),
                            "n_unique_configs": safe_unique(df[df[col] == r[col]], "condition_id"),
                        }
                    )
        if not full_df.empty:
            category_rows.append(
                {
                    "source": label,
                    "category_axis": "full_comsol_export",
                    "category": "full_comsol_rows",
                    "n_rows": len(full_df),
                    "n_unique_configs": safe_unique(full_df, "condition_id"),
                }
            )

    if not clinical.empty:
        malignant_terms = {
            "bcc",
            "scc_invasive",
            "scc_in_situ",
            "melanoma_in_situ",
            "melanoma_invasive_or_unspecified",
            "other_malignant",
        }
        clinical_tmp = clinical.copy()
        clinical_tmp["clinical_superclass"] = clinical_tmp["canonical_term"].apply(
            lambda x: "malignant_or_premalignant" if x in malignant_terms or "actinic" in str(x) else "benign_or_control"
        )
        for col in ["dataset", "canonical_term", "clinical_superclass"]:
            if col == "canonical_term":
                grouped = clinical_tmp.groupby(col, dropna=False)["n"].sum().reset_index(name="n_rows")
            else:
                grouped = clinical_tmp.groupby(col, dropna=False)["n"].sum().reset_index(name="n_rows")
            for _, r in grouped.iterrows():
                category_rows.append(
                    {
                        "source": "clinical_ontology",
                        "category_axis": col,
                        "category": r[col],
                        "n_rows": int(r["n_rows"]),
                        "n_unique_configs": None,
                    }
                )

    category_df = pd.DataFrame(category_rows)

    coverage = {
        "mmp_manifest_rows": len(fig3),
        "mmp_unique_configs": safe_unique(fig3, "case_source_group"),
        "mmp_trainable_cases_qc": fig3_qc.get("n_trainable_cases"),
        "mmp_method1_2d_cases_qc": fig3_qc.get("n_method1_cases"),
        "mmp_method2_3d_cases_qc": fig3_qc.get("n_method2_cases"),
        "mmp_duplicate_hash_cases_qc": fig3_qc.get("n_duplicate_hash_cases"),
        "mmp_strain_monotonicity_fail_qc": fig3_qc.get("n_strain_monotonicity_fail"),
        "rigidity_3d_rows": len(rigidity),
        "rigidity_3d_depth_levels": safe_unique(rigidity, "depth_mm"),
        "rigidity_3d_depth_values": safe_values(rigidity, "depth_mm"),
        "rigidity_3d_strain_levels": safe_unique(rigidity, "strain_float"),
        "rigidity_3d_E_levels": safe_unique(rigidity, "E_tag"),
        "piezo_case_count": piezo_summary.get("case_count") or len(piezo),
        "piezo_condition_count": piezo_summary.get("condition_count") or safe_unique(piezo, "condition_id"),
        "piezo_full_rows": len(piezo_full),
        "ultrasound_case_count": us_summary.get("case_count") or len(us),
        "ultrasound_condition_count": us_summary.get("condition_count") or safe_unique(us, "condition_id"),
        "ultrasound_full_rows": len(us_full),
        "clinical_total": int(clinical["n"].sum()) if "n" in clinical.columns and not clinical.empty else None,
        "clinical_dataset_count": safe_unique(clinical, "dataset"),
        "clinical_canonical_term_count": safe_unique(clinical, "canonical_term"),
        "figure2_public_projection_n": fig2_support.get("panel_f", {}).get("projection_n"),
        "figure2_public_target_n": fig2_support.get("panel_f", {}).get("target_n"),
        "figure2_public_overlap_fraction": fig2_support.get("panel_f", {}).get("target_public_overlap_fraction"),
    }
    return coverage, inventory_df, category_df


def make_experiment_design(coverage: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fig2_rows = [
        {
            "figure": "Figure 2",
            "target": "2D apparent mechanics posterior p(E_app(x,y), boundary, contrast | y_d, d, q_d)",
            "axis": "operator/readout",
            "recommended_classes": "MMP, PIEZO_PASSIVE, ULTRASOUND_STRAIN",
            "n_classes": 3,
            "minimum_per_class": ">= 500 train configs for primary MMP; >= 150 simulated benchmark readouts for piezo/US as operator-shift tests",
            "current_reference_support": f"MMP manifest rows={coverage.get('mmp_manifest_rows')}, trainable cases={coverage.get('mmp_trainable_cases_qc')}; piezo={coverage.get('piezo_case_count')}; US={coverage.get('ultrasound_case_count')}",
            "role_in_domain_randomization": "operator randomization: different forward readout kernels and noise/contact modes",
        },
        {
            "figure": "Figure 2",
            "target": "2D apparent mechanics posterior p(E_app(x,y), boundary, contrast | y_d, d, q_d)",
            "axis": "morphology/source",
            "recommended_classes": "synthetic geometric primitives, clinical lesion priors, phantom anchors",
            "n_classes": 3,
            "minimum_per_class": ">= 200 configs/class for learning curve; phantom anchors can be smaller but must be reserved for support coverage",
            "current_reference_support": f"clinical support: {coverage.get('clinical_dataset_count')} datasets, {coverage.get('clinical_canonical_term_count')} canonical terms; public projection n={coverage.get('figure2_public_projection_n')}, target n={coverage.get('figure2_public_target_n')}",
            "role_in_domain_randomization": "morphology randomization: shape, area, boundary roughness, lesion priors",
        },
        {
            "figure": "Figure 2",
            "target": "2D apparent mechanics posterior p(E_app(x,y), boundary, contrast | y_d, d, q_d)",
            "axis": "contrast",
            "recommended_classes": "soft/hypoelastic, matched-low-contrast, stiff/hyperelastic",
            "n_classes": 3,
            "minimum_per_class": ">= 500 train configs/class or balanced by inverse-frequency sampling",
            "current_reference_support": "3D rigidity scalar labels expose log contrast ck/log_ck; piezo/US manifests expose A_contrast and H_heterogeneity",
            "role_in_domain_randomization": "mechanical contrast randomization and low-observability stress test",
        },
        {
            "figure": "Figure 2",
            "target": "2D apparent mechanics posterior p(E_app(x,y), boundary, contrast | y_d, d, q_d)",
            "axis": "noise/contact/calibration",
            "recommended_classes": "clean, low-noise/calibration-shift, high-noise/contact-shift",
            "n_classes": 3,
            "minimum_per_class": ">= 3 stochastic seeds per config; keep all noise siblings inside same split",
            "current_reference_support": "MMP strain/readout rows plus piezo/US bonded/tilted and u_load/compression levels",
            "role_in_domain_randomization": "measurement-process randomization without truth leakage",
        },
    ]

    fig3_rows = [
        {
            "figure": "Figure 3",
            "target": "3D latent mechanics posterior p(E(x,y,z), z_top, z_bottom, thickness | y_d, d, q_d)",
            "axis": "operator/readout",
            "recommended_classes": "MMP, PIEZO_PASSIVE, ULTRASOUND_STRAIN",
            "n_classes": 3,
            "minimum_per_class": ">= 1000 MMP configs for primary posterior; piezo/US at least existing 150 readouts/operator for observability benchmark, expand to >= 500/operator for final learning-curve claim",
            "current_reference_support": f"MMP trainable={coverage.get('mmp_trainable_cases_qc')}; piezo={coverage.get('piezo_case_count')}; US={coverage.get('ultrasound_case_count')}",
            "role_in_domain_randomization": "operator observability: MMP/US should constrain depth better; piezo should show wider or multimodal posterior in deep/low-observability cases",
        },
        {
            "figure": "Figure 3",
            "target": "3D latent mechanics posterior p(E(x,y,z), z_top, z_bottom, thickness | y_d, d, q_d)",
            "axis": "depth",
            "recommended_classes": "shallow 0-0.75 mm, mid 0.75-1.5 mm, deep 1.5-3 mm, OOD-deep >3 mm",
            "n_classes": 4,
            "minimum_per_class": "ID bins: >= 300 train configs/bin; OOD-deep: held-out only, >= 100 configs for failure utility",
            "current_reference_support": f"3D rigidity depth levels={coverage.get('rigidity_3d_depth_levels')} values={coverage.get('rigidity_3d_depth_values')}",
            "role_in_domain_randomization": "depth randomization plus explicit OOD depth holdout",
        },
        {
            "figure": "Figure 3",
            "target": "3D latent mechanics posterior p(E(x,y,z), z_top, z_bottom, thickness | y_d, d, q_d)",
            "axis": "thickness",
            "recommended_classes": "thin, medium, thick",
            "n_classes": 3,
            "minimum_per_class": ">= 300 train configs/class, stratified inside depth bins",
            "current_reference_support": "Current manifest has depth priors and H_heterogeneity; explicit z_top/thickness truth should be normalized into depth_truth.csv",
            "role_in_domain_randomization": "thickness randomization separates depth vs volume effects",
        },
        {
            "figure": "Figure 3",
            "target": "3D latent mechanics posterior p(E(x,y,z), z_top, z_bottom, thickness | y_d, d, q_d)",
            "axis": "contrast",
            "recommended_classes": "soft/hypoelastic, matched-low-contrast, stiff/hyperelastic",
            "n_classes": 3,
            "minimum_per_class": ">= 300 train configs/class; oversample low-contrast because it drives multimodal posterior",
            "current_reference_support": "E_tag/ck/log_ck in 3D rigidity mapping; A_contrast in piezo/US/MMP manifests",
            "role_in_domain_randomization": "contrast randomization and posterior ambiguity stress test",
        },
        {
            "figure": "Figure 3",
            "target": "3D latent mechanics posterior p(E(x,y,z), z_top, z_bottom, thickness | y_d, d, q_d)",
            "axis": "contact/load/noise",
            "recommended_classes": "baseline, mild shift, hard shift",
            "n_classes": 3,
            "minimum_per_class": ">= 3 seeds/config; contact/load siblings must share config split",
            "current_reference_support": f"MMP strain levels={coverage.get('rigidity_3d_strain_levels')}; piezo/US use 5 u_load/compression levels and bonded/tilted contact",
            "role_in_domain_randomization": "operator metadata q_d randomization and robustness gate",
        },
    ]

    experiment_design = pd.DataFrame(fig2_rows + fig3_rows)

    learning_curve = pd.DataFrame(
        [
            {
                "figure": "Figure 2",
                "N_grid": "500,1000,2000,5000,10000,25000,50000",
                "models": "U-Net/direct regressor; vanilla diffusion; DPS-style inverse diffusion; operator-conditioned diffusion",
                "seeds": "0,1,2",
                "primary_metrics": "MAE(E_app), RMSE(E_app), SSIM, Boundary-F1, CRPS, forward residual",
                "data_efficiency_outputs": "N90, N95, AULC, marginal_gain_last_doubling, seed_SD, DER=N90_baseline/N90_ours",
            },
            {
                "figure": "Figure 3",
                "N_grid": "1000,2000,5000,10000,25000,50000,100000",
                "models": "direct depth regressor; vanilla 3D diffusion; DPS-style inverse diffusion; operator-conditioned 3D diffusion",
                "seeds": "0,1,2",
                "primary_metrics": "z_bottom MAE/RMSE/P90AE, CRPS/WIS, Cov50/80/90/95, Width90, interval IoU, AUROC_fail, multimodality frequency",
                "data_efficiency_outputs": "N90, N95, AULC, marginal_gain_last_doubling, seed_SD, DER_CRPS, DER_Cal",
            },
        ]
    )

    metric_gate = pd.DataFrame(
        [
            {
                "gate": "G0 inventory/leakage",
                "applies_to": "Figure 2 and Figure 3",
                "pass_threshold": "No shared config_id across train/val/test; no repeated latent target across splits under only noise changes; no truth fields inside operator metadata q_d; no test/real normalization leakage",
                "yellow_threshold": "Provenance incomplete but no detected split leakage",
                "fail_threshold": "Any config/latent target leakage or posterior derived from truth",
                "outputs": "inventory CSV, split audit, duplicate hash audit",
            },
            {
                "gate": "G1 simulator support",
                "applies_to": "Figure 2 and Figure 3",
                "pass_threshold": ">=90% phantom/animal/ex vivo MMP readouts fall inside 95% simulated feature envelope; two-sample AUC near chance after matching",
                "yellow_threshold": "75-90% inside support or phantom-only pass",
                "fail_threshold": "<75% inside support",
                "outputs": "kNN/Mahalanobis/MMD/conformal p-value support report",
            },
            {
                "gate": "G2 learning plateau",
                "applies_to": "Figure 2 and Figure 3",
                "pass_threshold": "Last-doubling improvement <3-5%; current library >= estimated N90; seed SD <10% of mean",
                "yellow_threshold": "Plateau in point metrics but not CRPS/calibration",
                "fail_threshold": "Primary metrics still improve strongly at largest N",
                "outputs": "learning_curve_fits.csv, N90/N95/AULC",
            },
            {
                "gate": "G3 method-level data efficiency",
                "applies_to": "Figure 2 and Figure 3",
                "pass_threshold": "DER_MAE or DER_CRPS >=2 versus strongest baseline, or same-N CRPS/calibration/failure utility clearly better",
                "yellow_threshold": "Full-data performance best but N90 advantage <2x",
                "fail_threshold": "No stable advantage in low-data or full-data regimes",
                "outputs": "DER table and confidence intervals",
            },
            {
                "gate": "G4 Figure 2 reconstruction",
                "applies_to": "Figure 2",
                "pass_threshold": "MAE/SSIM/Boundary-F1/CRPS plateau; rejecting top 20% uncertainty reduces MAE or CRPS >=25%",
                "yellow_threshold": "Good maps but calibration weak",
                "fail_threshold": "Boundary or support-distance errors not controlled",
                "outputs": "fig2_metrics_by_N.csv",
            },
            {
                "gate": "G5 Figure 3 posterior",
                "applies_to": "Figure 3",
                "pass_threshold": "In-support MMP/US z_bottom MAE <=0.5-0.75 mm, P90AE <=1.5 mm, CRPS improves >=15-25%, interval IoU >=0.55-0.60; piezo deep cases show appropriately broad/multimodal posterior",
                "yellow_threshold": "Good point depth but posterior too conservative",
                "fail_threshold": "Depth posterior not calibrated or operator observability contradicted",
                "outputs": "fig3_metrics_by_N.csv",
            },
            {
                "gate": "G6 calibration/sharpness",
                "applies_to": "Figure 2 and Figure 3",
                "pass_threshold": "|Cov90-0.90| <=0.03 ID or <=0.05 hard/OOD; Width90 not inflated; PIT not strongly U-shaped/inverse-U; UCE/ENCE improves",
                "yellow_threshold": "Coverage fixed only by broadening interval",
                "fail_threshold": "Cov90 <0.80 or >0.98 with wide intervals",
                "outputs": "calibration_metrics.csv",
            },
            {
                "gate": "G7 OOD/failure utility",
                "applies_to": "Figure 2 and Figure 3",
                "pass_threshold": "AUROC_fail >=0.75-0.80; AUPRC >=2x prevalence; risk at 80% retained decreases; AUSE improves >=15-20%",
                "yellow_threshold": "Uncertainty increases with OOD but failure AUROC weak",
                "fail_threshold": "Uncertainty unrelated to high-error cases",
                "outputs": "ood_failure_metrics.csv",
            },
            {
                "gate": "G8 projection consistency",
                "applies_to": "Merge Figure 2 and Figure 3",
                "pass_threshold": "NMAE(E_app_2D, P_z[E_xyz]) <=0.10-0.15; SSIM >=0.80-0.85; centroid shift <=0.5-1.0 mm; area difference <=15-20%",
                "yellow_threshold": "Map projection consistent but boundary shifts",
                "fail_threshold": "2D and 3D posteriors describe inconsistent biomechanical states",
                "outputs": "projection_consistency_metrics.csv",
            },
        ]
    )

    domain_matrix = pd.DataFrame(
        [
            {
                "dimension": "operator",
                "classes": "MMP; PIEZO_PASSIVE; ULTRASOUND_STRAIN",
                "minimum_classes_for_claim": 3,
                "current_support": f"MMP trainable {coverage.get('mmp_trainable_cases_qc')}; piezo {coverage.get('piezo_case_count')}; US {coverage.get('ultrasound_case_count')}",
                "domain_randomization_goal": "Treat real MMP/phantom/animal/ex vivo readout as a variation inside simulated operator-feature support; piezo/US are simulated cross-operator benchmarks unless paired real data exist.",
            },
            {
                "dimension": "morphology",
                "classes": "geometric primitive; clinical lesion prior; phantom anchor",
                "minimum_classes_for_claim": 3,
                "current_support": f"clinical datasets {coverage.get('clinical_dataset_count')}; canonical terms {coverage.get('clinical_canonical_term_count')}",
                "domain_randomization_goal": "Cover lesion size, irregularity, roughness, and public/clinical priors rather than only idealized circles.",
            },
            {
                "dimension": "depth",
                "classes": "shallow; mid; deep; OOD-deep",
                "minimum_classes_for_claim": 4,
                "current_support": f"3D rigidity depth levels {coverage.get('rigidity_3d_depth_levels')}",
                "domain_randomization_goal": "ID bins support posterior learning; OOD-deep is reserved for failure utility and observability limits.",
            },
            {
                "dimension": "contrast",
                "classes": "soft/hypoelastic; matched-low; stiff/hyperelastic",
                "minimum_classes_for_claim": 3,
                "current_support": "A_contrast/H_heterogeneity and ck/log_ck available in reference manifests",
                "domain_randomization_goal": "Low contrast should be deliberately included because it creates non-identifiable/multimodal posteriors.",
            },
            {
                "dimension": "contact/load/noise",
                "classes": "baseline; mild shift; hard shift",
                "minimum_classes_for_claim": 3,
                "current_support": "piezo/US bonded/tilted x five load/compression levels; MMP has multi-strain Bz rows",
                "domain_randomization_goal": "Make operator metadata q_d variation explicit while preventing split leakage across noise/load siblings.",
            },
        ]
    )

    class_budget = pd.DataFrame(
        [
            {
                "figure": "Figure 2",
                "axis": "operator/readout",
                "class_label": "MMP soft magnetoelastic microneedle patch",
                "current_support_n": coverage.get("mmp_trainable_cases_qc"),
                "target_min_n": 500,
                "unit": "unique config_id for primary learning curve",
                "status": "YELLOW - close, but expand or justify with plateau",
            },
            {
                "figure": "Figure 2",
                "axis": "operator/readout",
                "class_label": "sheet piezoelectric device",
                "current_support_n": coverage.get("piezo_case_count"),
                "target_min_n": 150,
                "unit": "simulated benchmark readout, operator-shift only",
                "status": "PASS for benchmark; expand to 500 for final data-efficiency claim",
            },
            {
                "figure": "Figure 2",
                "axis": "operator/readout",
                "class_label": "ultrasound strain device",
                "current_support_n": coverage.get("ultrasound_case_count"),
                "target_min_n": 150,
                "unit": "simulated benchmark readout, operator-shift only",
                "status": "PASS for benchmark; expand to 500 for final data-efficiency claim",
            },
            {
                "figure": "Figure 2",
                "axis": "morphology/source",
                "class_label": "synthetic/geometric simulation",
                "current_support_n": coverage.get("rigidity_3d_rows"),
                "target_min_n": 200,
                "unit": "configs",
                "status": "PASS as scaffold; add more irregular boundaries for manuscript-strength coverage",
            },
            {
                "figure": "Figure 2",
                "axis": "morphology/source",
                "class_label": "clinical lesion priors",
                "current_support_n": coverage.get("clinical_total"),
                "target_min_n": 500,
                "unit": "clinical labels/priors; not paired MMP truth",
                "status": "PASS for prior/morphology support",
            },
            {
                "figure": "Figure 2",
                "axis": "morphology/source",
                "class_label": "real MMP phantom/animal/ex vivo anchors",
                "current_support_n": "not found in inspected manifests",
                "target_min_n": ">=20 phantom plus all available animal/ex vivo",
                "unit": "real readouts reserved for support coverage",
                "status": "MISSING for strong sim-to-real support gate",
            },
            {
                "figure": "Figure 2",
                "axis": "contrast",
                "class_label": "soft / hypoelastic",
                "current_support_n": "available through A_contrast/log_ck bins",
                "target_min_n": 500,
                "unit": "configs or balanced sampled rows",
                "status": "NEEDS explicit binning in simulator_params_2d.csv",
            },
            {
                "figure": "Figure 2",
                "axis": "contrast",
                "class_label": "matched / low contrast",
                "current_support_n": "available through A_contrast/log_ck bins",
                "target_min_n": 500,
                "unit": "configs or balanced sampled rows",
                "status": "NEEDS explicit binning; oversample for posterior ambiguity",
            },
            {
                "figure": "Figure 2",
                "axis": "contrast",
                "class_label": "stiff / hyperelastic",
                "current_support_n": "available through A_contrast/log_ck bins",
                "target_min_n": 500,
                "unit": "configs or balanced sampled rows",
                "status": "NEEDS explicit binning in simulator_params_2d.csv",
            },
            {
                "figure": "Figure 3",
                "axis": "operator/readout",
                "class_label": "MMP soft magnetoelastic microneedle patch",
                "current_support_n": coverage.get("mmp_method2_3d_cases_qc"),
                "target_min_n": 1000,
                "unit": "3D configs for primary posterior; use 445 MMP trainable as near-term scaffold",
                "status": "YELLOW/FAIL for full Figure 3 data-efficiency claim unless 3D cache expands or uses posterior surrogate",
            },
            {
                "figure": "Figure 3",
                "axis": "operator/readout",
                "class_label": "sheet piezoelectric device",
                "current_support_n": coverage.get("piezo_case_count"),
                "target_min_n": 150,
                "unit": "simulated benchmark readout",
                "status": "PASS for observability benchmark; do not claim real piezo",
            },
            {
                "figure": "Figure 3",
                "axis": "operator/readout",
                "class_label": "ultrasound strain device",
                "current_support_n": coverage.get("ultrasound_case_count"),
                "target_min_n": 150,
                "unit": "simulated benchmark readout",
                "status": "PASS for observability benchmark; do not claim real ultrasound",
            },
            {
                "figure": "Figure 3",
                "axis": "depth",
                "class_label": "shallow 0-0.75 mm",
                "current_support_n": 60,
                "target_min_n": 300,
                "unit": "3D rigidity rows/configs before augmentation",
                "status": "YELLOW - enough for scaffold, not final N90",
            },
            {
                "figure": "Figure 3",
                "axis": "depth",
                "class_label": "mid 0.75-1.5 mm",
                "current_support_n": 58,
                "target_min_n": 300,
                "unit": "3D rigidity rows/configs before augmentation",
                "status": "YELLOW - enough for scaffold, not final N90",
            },
            {
                "figure": "Figure 3",
                "axis": "depth",
                "class_label": "deep 1.5-3 mm",
                "current_support_n": 60,
                "target_min_n": 300,
                "unit": "3D rigidity rows/configs before augmentation",
                "status": "YELLOW - enough for scaffold, not final N90",
            },
            {
                "figure": "Figure 3",
                "axis": "depth",
                "class_label": "OOD-deep >3 mm",
                "current_support_n": 120,
                "target_min_n": 100,
                "unit": "held-out OOD/failure configs",
                "status": "PASS for OOD utility, keep out of ID training",
            },
            {
                "figure": "Figure 3",
                "axis": "thickness",
                "class_label": "thin",
                "current_support_n": "not explicit as z_top/thickness truth",
                "target_min_n": 300,
                "unit": "configs",
                "status": "MISSING until depth_truth.csv has z_top and thickness",
            },
            {
                "figure": "Figure 3",
                "axis": "thickness",
                "class_label": "medium",
                "current_support_n": "not explicit as z_top/thickness truth",
                "target_min_n": 300,
                "unit": "configs",
                "status": "MISSING until depth_truth.csv has z_top and thickness",
            },
            {
                "figure": "Figure 3",
                "axis": "thickness",
                "class_label": "thick",
                "current_support_n": "not explicit as z_top/thickness truth",
                "target_min_n": 300,
                "unit": "configs",
                "status": "MISSING until depth_truth.csv has z_top and thickness",
            },
            {
                "figure": "Figure 3",
                "axis": "contrast",
                "class_label": "soft / hypoelastic",
                "current_support_n": "available through E_tag/log_ck bins",
                "target_min_n": 300,
                "unit": "configs",
                "status": "NEEDS explicit binning and balance",
            },
            {
                "figure": "Figure 3",
                "axis": "contrast",
                "class_label": "matched / low contrast",
                "current_support_n": "available through E_tag/log_ck bins",
                "target_min_n": 300,
                "unit": "configs",
                "status": "NEEDS explicit binning; oversample for ambiguity",
            },
            {
                "figure": "Figure 3",
                "axis": "contrast",
                "class_label": "stiff / hyperelastic",
                "current_support_n": "available through E_tag/log_ck bins",
                "target_min_n": 300,
                "unit": "configs",
                "status": "NEEDS explicit binning and balance",
            },
            {
                "figure": "Figure 3",
                "axis": "contact/load/noise",
                "class_label": "baseline",
                "current_support_n": "MMP S=0; piezo/US u=0",
                "target_min_n": ">=3 seeds/config",
                "unit": "noise/load sibling rows",
                "status": "AVAILABLE; enforce same split by config_id",
            },
            {
                "figure": "Figure 3",
                "axis": "contact/load/noise",
                "class_label": "mild shift",
                "current_support_n": "MMP low strains; piezo/US u=0.01/0.02",
                "target_min_n": ">=3 seeds/config",
                "unit": "noise/load sibling rows",
                "status": "AVAILABLE; enforce same split by config_id",
            },
            {
                "figure": "Figure 3",
                "axis": "contact/load/noise",
                "class_label": "hard shift",
                "current_support_n": "MMP high strains; piezo/US u=0.04/0.08 and tilted",
                "target_min_n": ">=3 seeds/config",
                "unit": "noise/load sibling rows",
                "status": "AVAILABLE; enforce same split by config_id",
            },
        ]
    )
    return experiment_design, learning_curve, metric_gate, domain_matrix, class_budget


def write_report(
    coverage: dict[str, Any],
    inventory: pd.DataFrame,
    category: pd.DataFrame,
    experiment_design: pd.DataFrame,
    learning_curve: pd.DataFrame,
    metric_gate: pd.DataFrame,
    domain_matrix: pd.DataFrame,
) -> None:
    summary_lines = [
        "# Figure 2-3 Data Efficiency Experiment Design",
        "",
        "This report is based on `data efficiency.pdf` pages 25-68 and the local reference data folders. It is an experiment design and readiness plan, not a claim that the final model has already passed all gates.",
        "",
        "## Core Decision",
        "",
        "Use two linked but separate data-efficiency experiments:",
        "",
        "1. Figure 2: 2D apparent mechanics, `E_app(x,y)`, focused on map, boundary, contrast, posterior calibration, and real-MMP support coverage.",
        "2. Figure 3: 3D latent mechanics, `E(x,y,z)` / `z_bottom`, focused on continuous depth posterior, operator observability, calibration/sharpness, multimodality, and OOD/failure utility.",
        "",
        "Do not frame domain randomization as 'large simulated N'. The gate is whether real or held-out readouts fall inside the simulated readout-support manifold and whether learning curves are saturated.",
        "",
        "## Existing Reference Coverage",
        "",
        f"- MMP/MN manifest rows: {coverage.get('mmp_manifest_rows')}; QC trainable cases: {coverage.get('mmp_trainable_cases_qc')}; method1 2D cases: {coverage.get('mmp_method1_2d_cases_qc')}; method2 3D cases: {coverage.get('mmp_method2_3d_cases_qc')}.",
        f"- 3D rigidity mapping rows: {coverage.get('rigidity_3d_rows')}; depth levels: {coverage.get('rigidity_3d_depth_levels')} ({coverage.get('rigidity_3d_depth_values')}); strain levels: {coverage.get('rigidity_3d_strain_levels')}.",
        f"- Piezo benchmark: {coverage.get('piezo_case_count')} readout cases across {coverage.get('piezo_condition_count')} conditions; full COMSOL rows: {coverage.get('piezo_full_rows')}.",
        f"- Ultrasound benchmark: {coverage.get('ultrasound_case_count')} readout cases across {coverage.get('ultrasound_condition_count')} conditions; full COMSOL rows: {coverage.get('ultrasound_full_rows')}.",
        f"- Clinical ontology support: {coverage.get('clinical_dataset_count')} datasets, {coverage.get('clinical_canonical_term_count')} canonical terms, total n={coverage.get('clinical_total')}.",
        "",
        "Important caution: piezo and ultrasound are simulation benchmarks here unless paired real piezo/US files are added. Real-support claims should be anchored on MMP phantom/animal/ex vivo readouts.",
        "",
        "## Recommended Category Counts",
        "",
        "- Figure 2 should include 4 category axes and 12 named classes: operator/readout (3), morphology/source (3), contrast (3), noise/contact/calibration (3).",
        "- Figure 3 should include 5 category axes and 16 named classes: operator/readout (3), depth (4), thickness (3), contrast (3), contact/load/noise (3).",
        "- For a strong domain-randomization claim, keep each primary ID stratum represented by at least 3 random seeds and split by `config_id`, not by readout row.",
        "",
        "## Learning-Curve Rule",
        "",
        "Fit `m(N)=m_inf+a*N^-alpha` for lower-is-better metrics and `m(N)=m_inf-a*N^-alpha` for higher-is-better metrics. Report N90, N95, AULC, marginal gain after last doubling, seed SD, and DER = N90(baseline) / N90(ours).",
        "",
        "## Gate Summary",
        "",
        "A figure is ready only if inventory/leakage, simulator support, learning plateau, method-level data efficiency, calibration/sharpness, and OOD/failure utility are at least PASS/YELLOW with no FAIL. Figure 3 additionally requires a continuous depth posterior gate; merging Figure 2 and Figure 3 additionally requires projection consistency.",
        "",
        "## Output Files",
        "",
        "- `fig23_existing_data_inventory.csv`: source availability and high-level counts.",
        "- `fig23_reference_category_counts.csv`: observed categories/counts from local manifests.",
        "- `fig23_experiment_design.csv`: recommended Figure 2/3 category plan.",
        "- `fig23_recommended_class_budget.csv`: direct per-class current support and target sample budget.",
        "- `fig23_learning_curve_plan.csv`: N grids, models, metrics.",
        "- `fig23_metric_gate_table.csv`: PASS/YELLOW/FAIL thresholds.",
        "- `fig23_domain_randomization_matrix.csv`: domain randomization dimensions and minimum class counts.",
        "- `fig23_counts_from_reference_data.json`: compact JSON coverage summary.",
    ]
    (OUT / "figure23_data_efficiency_design_report.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    coverage, inventory, category = summarize_reference_data()
    experiment_design, learning_curve, metric_gate, domain_matrix, class_budget = make_experiment_design(coverage)

    inventory.to_csv(OUT / "fig23_existing_data_inventory.csv", index=False, encoding="utf-8-sig")
    category.to_csv(OUT / "fig23_reference_category_counts.csv", index=False, encoding="utf-8-sig")
    experiment_design.to_csv(OUT / "fig23_experiment_design.csv", index=False, encoding="utf-8-sig")
    class_budget.to_csv(OUT / "fig23_recommended_class_budget.csv", index=False, encoding="utf-8-sig")
    learning_curve.to_csv(OUT / "fig23_learning_curve_plan.csv", index=False, encoding="utf-8-sig")
    metric_gate.to_csv(OUT / "fig23_metric_gate_table.csv", index=False, encoding="utf-8-sig")
    domain_matrix.to_csv(OUT / "fig23_domain_randomization_matrix.csv", index=False, encoding="utf-8-sig")
    (OUT / "fig23_counts_from_reference_data.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(coverage, inventory, category, experiment_design, learning_curve, metric_gate, domain_matrix)

    print(json.dumps({"out": str(OUT), "coverage": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
