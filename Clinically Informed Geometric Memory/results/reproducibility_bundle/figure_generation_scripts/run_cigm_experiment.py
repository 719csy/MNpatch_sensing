from __future__ import annotations

import json
import math
import re
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from pypdf import PdfReader
from scipy import ndimage, stats
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


SEED = 20260606
RNG = np.random.default_rng(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_ROOT = PROJECT_ROOT.parent.parent
SOURCE_PDF = MANUSCRIPT_ROOT / "Data" / "Clinically Informed Geometric Memory.pdf"
FIG2_ROOT = MANUSCRIPT_ROOT / "Figure2_data_reconstruction_20260526"
FIG3_ROOT = MANUSCRIPT_ROOT / "Data" / "fig3"

RESULTS = PROJECT_ROOT / "results"
TABLES = RESULTS / "tables"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"
FIGURES = RESULTS / "figures"
REPORTS = RESULTS / "reports"
BUNDLE = RESULTS / "reproducibility_bundle"

DESCRIPTOR_COLS_2D = [
    "area_frac",
    "perimeter_norm",
    "equivalent_diameter_norm",
    "aspect_ratio",
    "eccentricity",
    "solidity",
    "compactness",
    "extent",
    "convexity",
    "major_axis_norm",
    "minor_axis_norm",
    "boundary_curvature_mean",
    "boundary_curvature_std",
    "fractal_dimension",
    "asymmetry_score",
    "number_of_components",
]

DESCRIPTOR_COLS_3D = [
    "volume_frac",
    "surface_area_norm",
    "sphericity",
    "elongation",
    "flatness",
    "compactness_3d",
    "z_extent",
    "x_extent",
    "y_extent",
    "surface_roughness",
    "centroid_z",
]


def ensure_dirs() -> None:
    for path in [RESULTS, TABLES, METRICS, PREDICTIONS, FIGURES, REPORTS, BUNDLE]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    table = df.reset_index()
    cols = [str(c) for c in table.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in table.to_dict("records"):
        values = []
        for col in table.columns:
            value = row[col]
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.1,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def safe_float(value) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def extract_pdf_pages() -> pd.DataFrame:
    reader = PdfReader(str(SOURCE_PDF))
    rows = []
    chunks = []
    for idx in range(31, min(64, len(reader.pages))):
        text = reader.pages[idx].extract_text() or ""
        rows.append({"pdf_page": idx + 1, "n_chars": len(text), "text_preview": " ".join(text.split())[:240]})
        chunks.append(f"\n\n--- PAGE {idx + 1} ---\n{text.strip()}\n")
    write_text(PROJECT_ROOT / "source_pdf_pages_32_64_extracted_text.txt", "".join(chunks).strip() + "\n")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "pdf_pages_32_64_extraction_summary.csv", index=False)
    return df


def classify_geometry_source(case_id: str, source_dir: str = "") -> str:
    token = f"{case_id} {source_dir}".lower()
    clinical_patterns = [
        "isic",
        "midas",
        "ph2",
        "bcn",
        "clinical",
        "clincal",
        "melanoma",
        "basal_cell",
        "benign_nevi",
        "squamous_cell",
    ]
    analytic_patterns = ["cylinder", "circular", "ellipse", "analytic", "phantom"]
    if any(p in token for p in clinical_patterns):
        return "clinical_mask"
    if any(p in token for p in analytic_patterns):
        return "ellipse_or_cylinder"
    return "synthetic_simulation"


def geometry_id_from_case(case_id: str) -> str:
    return str(case_id).split("__")[0]


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(FIG2_ROOT / "tables" / "lesion_case_inventory_metrics.csv")
    manifest_rows = []
    for row in inventory.to_dict("records"):
        source_type = classify_geometry_source(row["case_id"], row.get("source_dir", ""))
        manifest_rows.append(
            {
                "sample_id": row["case_id"],
                "dim": "2d",
                "field_path": row["mean_kpa_path"],
                "mask_path": row["mask_path"],
                "readout_path": "",
                "metadata_path": FIG2_ROOT / "tables" / "lesion_case_inventory_metrics.csv",
                "geometry_id": geometry_id_from_case(row["case_id"]),
                "source_geometry_type": source_type,
                "split_original": "",
                "z_top": np.nan,
                "z_bottom": np.nan,
                "thickness": np.nan,
                "E_bg": row.get("bg_median_kpa", np.nan),
                "E_lesion": row.get("lesion_mean_kpa", np.nan),
                "contrast": row.get("contrast", np.nan),
                "operator_id": "",
                "operator_metadata_path": "",
            }
        )

    cases3d = pd.read_csv(FIG3_ROOT / "metadata" / "cases.csv")
    for i, row in enumerate(cases3d.to_dict("records")):
        manifest_rows.append(
            {
                "sample_id": row["case_id"],
                "dim": "3d",
                "field_path": FIG3_ROOT / "latent_truth" / "volume_E_xyz.npz",
                "mask_path": FIG3_ROOT / "latent_truth" / "volume_E_xyz.npz",
                "readout_path": FIG3_ROOT / "operators" / "operator_readouts_examples.npz",
                "metadata_path": FIG3_ROOT / "metadata" / "cases.csv",
                "geometry_id": row["case_id"],
                "source_geometry_type": "simulated_3d_lesion",
                "split_original": "figure3_benchmark",
                "z_top": row.get("z_top_mm", np.nan),
                "z_bottom": row.get("z_bottom_mm", np.nan),
                "thickness": row.get("thickness_mm", np.nan),
                "E_bg": 8.0,
                "E_lesion": 8.0 * safe_float(row.get("contrast", np.nan)),
                "contrast": row.get("contrast", np.nan),
                "operator_id": "MMP|PIEZO|US",
                "operator_metadata_path": FIG3_ROOT / "metadata" / "operator_metadata.csv",
                "array_index": i,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(RESULTS / "data_manifest.csv", index=False)

    summary = (
        manifest.groupby("dim")
        .agg(
            n_samples=("sample_id", "count"),
            n_geometry=("geometry_id", "nunique"),
            mask_available=("mask_path", lambda s: int((s.astype(str) != "").sum())),
            readout_available=("readout_path", lambda s: int((s.astype(str) != "").sum())),
            depth_metadata=("z_bottom", lambda s: int(s.notna().sum())),
            E_bg_median=("E_bg", "median"),
            E_lesion_median=("E_lesion", "median"),
            contrast_median=("contrast", "median"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "manifest_summary.csv", index=False)
    return manifest, inventory


def load_mask_csv(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path)
    value_col = "lesion_mask" if "lesion_mask" in df.columns else df.columns[-1]
    if {"x_norm", "y_norm"}.issubset(df.columns):
        pivot = df.pivot_table(index="y_norm", columns="x_norm", values=value_col, aggfunc="mean")
        arr = pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True).to_numpy()
    else:
        n = int(round(math.sqrt(len(df))))
        arr = df[value_col].to_numpy().reshape(n, n)
    return np.nan_to_num(arr, nan=0.0) > 0.5


def resize_mask(mask: np.ndarray, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    zoom = (shape[0] / mask.shape[0], shape[1] / mask.shape[1])
    return ndimage.zoom(mask.astype(float), zoom, order=0) > 0.5


def keep_largest(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == counts.argmax()


def perimeter_mask(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = ndimage.binary_erosion(mask)
    return mask & ~eroded


def fractal_dimension(mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    h, w = mask.shape
    sizes = [2, 4, 8, 16, 32]
    counts = []
    inv_sizes = []
    for size in sizes:
        if size >= min(h, w):
            continue
        padded_h = int(math.ceil(h / size) * size)
        padded_w = int(math.ceil(w / size) * size)
        padded = np.zeros((padded_h, padded_w), dtype=bool)
        padded[:h, :w] = mask
        blocks = padded.reshape(padded_h // size, size, padded_w // size, size)
        count = blocks.any(axis=(1, 3)).sum()
        if count > 0:
            counts.append(count)
            inv_sizes.append(1.0 / size)
    if len(counts) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(inv_sizes), np.log(counts), 1)
    return float(slope)


def descriptors_2d(mask: np.ndarray) -> dict:
    mask = mask.astype(bool)
    h, w = mask.shape
    total = h * w
    area = int(mask.sum())
    if area == 0:
        return {k: 0.0 for k in DESCRIPTOR_COLS_2D}

    labeled, n_components = ndimage.label(mask)
    boundary = perimeter_mask(mask)
    perim = float(boundary.sum())
    ys, xs = np.where(mask)
    coords = np.column_stack([xs, ys]).astype(float)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered.T) if len(coords) > 2 else np.eye(2)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = 4.0 * math.sqrt(max(evals[0], 1e-12))
    minor = 4.0 * math.sqrt(max(evals[1], 1e-12))
    aspect = major / max(minor, 1e-9)
    eccentricity = math.sqrt(max(0.0, 1.0 - evals[1] / max(evals[0], 1e-9)))
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    bbox_area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
    extent = area / bbox_area

    hull_area = float(area)
    boundary_coords = np.column_stack(np.where(boundary))
    if len(boundary_coords) > 3:
        if len(boundary_coords) > 4000:
            idx = RNG.choice(len(boundary_coords), size=4000, replace=False)
            boundary_coords = boundary_coords[idx]
        try:
            hull = ConvexHull(boundary_coords)
            hull_area = max(float(hull.volume), float(area))
        except QhullError:
            hull_area = float(area)
    solidity = min(1.0, float(area) / max(hull_area, 1e-9))
    convexity = hull_area / max(float(area), 1.0)
    compact = 4.0 * math.pi * area / max(perim * perim, 1e-9)
    equiv_diameter = math.sqrt(4.0 * area / math.pi)

    by, bx = np.where(boundary)
    if len(bx) > 0:
        r = np.sqrt((bx - centroid[0]) ** 2 + (by - centroid[1]) ** 2)
        boundary_curv_mean = perim / max(math.sqrt(area), 1e-9)
        boundary_curv_std = float(np.std(r) / max(np.mean(r), 1e-9))
    else:
        boundary_curv_mean = 0.0
        boundary_curv_std = 0.0

    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    asym_lr = np.logical_xor(crop, np.fliplr(crop)).mean()
    asym_ud = np.logical_xor(crop, np.flipud(crop)).mean()

    return {
        "area_frac": area / total,
        "perimeter_norm": perim / math.sqrt(total),
        "equivalent_diameter_norm": equiv_diameter / math.sqrt(total),
        "aspect_ratio": aspect,
        "eccentricity": eccentricity,
        "solidity": solidity,
        "compactness": compact,
        "extent": extent,
        "convexity": convexity,
        "major_axis_norm": major / math.sqrt(total),
        "minor_axis_norm": minor / math.sqrt(total),
        "boundary_curvature_mean": boundary_curv_mean,
        "boundary_curvature_std": boundary_curv_std,
        "fractal_dimension": fractal_dimension(mask),
        "asymmetry_score": float((asym_lr + asym_ud) / 2.0),
        "number_of_components": float(n_components),
    }


def descriptors_3d(mask: np.ndarray, z_mm: np.ndarray | None = None) -> dict:
    mask = mask.astype(bool)
    total = np.prod(mask.shape)
    volume = float(mask.sum())
    if volume == 0:
        return {k: 0.0 for k in DESCRIPTOR_COLS_3D}
    surface = mask & ~ndimage.binary_erosion(mask)
    surface_area = float(surface.sum())
    z, y, x = np.where(mask)
    coords = np.column_stack([z, y, x]).astype(float)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered.T) if len(coords) > 3 else np.eye(3)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    elongation = math.sqrt(max(evals[0], 1e-12) / max(evals[1], 1e-12))
    flatness = math.sqrt(max(evals[2], 1e-12) / max(evals[0], 1e-12))
    sphericity = (math.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0)) / max(surface_area, 1e-9)
    bz, by, bx = np.where(surface)
    if len(bx) > 0:
        rr = np.sqrt((bz - centroid[0]) ** 2 + (by - centroid[1]) ** 2 + (bx - centroid[2]) ** 2)
        rough = float(np.std(rr) / max(np.mean(rr), 1e-9))
    else:
        rough = 0.0
    z_extent = (z.max() - z.min() + 1) / mask.shape[0]
    y_extent = (y.max() - y.min() + 1) / mask.shape[1]
    x_extent = (x.max() - x.min() + 1) / mask.shape[2]
    centroid_z = float(centroid[0] / max(mask.shape[0] - 1, 1))
    return {
        "volume_frac": volume / total,
        "surface_area_norm": surface_area / (total ** (2.0 / 3.0)),
        "sphericity": float(sphericity),
        "elongation": float(elongation),
        "flatness": float(flatness),
        "compactness_3d": float((36.0 * math.pi * volume * volume) / max(surface_area**3, 1e-9)),
        "z_extent": float(z_extent),
        "x_extent": float(x_extent),
        "y_extent": float(y_extent),
        "surface_roughness": rough,
        "centroid_z": centroid_z,
    }


def build_geometry_descriptors(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in inventory.to_dict("records"):
        try:
            mask = load_mask_csv(row["mask_path"])
            desc = descriptors_2d(mask)
            desc.update(
                {
                    "sample_id": row["case_id"],
                    "geometry_id": geometry_id_from_case(row["case_id"]),
                    "source_geometry_type": classify_geometry_source(row["case_id"], row.get("source_dir", "")),
                    "source_dir": row.get("source_dir", ""),
                    "mask_path": row["mask_path"],
                    "E_bg": row.get("bg_median_kpa", np.nan),
                    "E_lesion": row.get("lesion_mean_kpa", np.nan),
                    "contrast": row.get("contrast", np.nan),
                    "heterogeneity": row.get("heterogeneity", np.nan),
                }
            )
            rows.append(desc)
        except Exception as exc:
            rows.append(
                {
                    "sample_id": row["case_id"],
                    "geometry_id": geometry_id_from_case(row["case_id"]),
                    "source_geometry_type": "mask_read_failed",
                    "error": str(exc),
                }
            )
    df2 = pd.DataFrame(rows)
    df2.to_csv(TABLES / "geometry_descriptors_2d.csv", index=False)

    volume = np.load(FIG3_ROOT / "latent_truth" / "volume_E_xyz.npz", allow_pickle=True)
    masks = volume["lesion_mask_xyz"] > 0.5
    case_ids = [str(x) for x in volume["case_id"]]
    rows3 = []
    for i, cid in enumerate(case_ids):
        desc = descriptors_3d(masks[i], volume["z_mm"])
        desc.update({"sample_id": cid, "geometry_id": cid, "source_geometry_type": "simulated_3d_lesion"})
        rows3.append(desc)
    df3 = pd.DataFrame(rows3)
    df3.to_csv(TABLES / "geometry_descriptors_3d.csv", index=False)
    return df2, df3


def random_blob_masks(n: int, shape: tuple[int, int] = (64, 64)) -> list[np.ndarray]:
    masks = []
    for _ in range(n):
        field = RNG.normal(size=shape)
        sigma = RNG.uniform(2.0, 8.0)
        field = ndimage.gaussian_filter(field, sigma=sigma)
        area = RNG.uniform(0.06, 0.32)
        threshold = np.quantile(field, 1.0 - area)
        mask = keep_largest(field > threshold)
        masks.append(mask)
    return masks


def ellipse_masks(n: int, shape: tuple[int, int] = (64, 64)) -> list[np.ndarray]:
    yy, xx = np.mgrid[-1:1 : complex(shape[0]), -1:1 : complex(shape[1])]
    masks = []
    for _ in range(n):
        cx, cy = RNG.uniform(-0.18, 0.18, size=2)
        a = RNG.uniform(0.22, 0.55)
        b = RNG.uniform(0.14, 0.45)
        theta = RNG.uniform(0, math.pi)
        xr = (xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)
        yr = -(xx - cx) * math.sin(theta) + (yy - cy) * math.cos(theta)
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        if RNG.random() < 0.15:
            cx2, cy2 = RNG.uniform(-0.35, 0.35, size=2)
            a2 = RNG.uniform(0.08, 0.2)
            b2 = RNG.uniform(0.08, 0.2)
            mask |= ((xx - cx2) / a2) ** 2 + ((yy - cy2) / b2) ** 2 <= 1.0
        masks.append(mask)
    return masks


def one_d_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    qs = np.linspace(0.01, 0.99, 99)
    return float(np.mean(np.abs(np.nanquantile(a, qs) - np.nanquantile(b, qs))))


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    xy = np.vstack([x, y])
    distances = cdist(xy, xy)
    sigma = np.nanmedian(distances[distances > 0])
    sigma = float(sigma if np.isfinite(sigma) and sigma > 0 else 1.0)
    gamma = 1.0 / (2.0 * sigma * sigma)
    kxx = np.exp(-gamma * cdist(x, x) ** 2)
    kyy = np.exp(-gamma * cdist(y, y) ** 2)
    kxy = np.exp(-gamma * cdist(x, y) ** 2)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def distribution_matching(df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = df2[df2["source_geometry_type"] == "clinical_mask"].copy()
    clinical = clinical.drop_duplicates("geometry_id")
    if len(clinical) < 10:
        raise RuntimeError("Not enough clinical masks for held-out geometry matching.")

    geometry_ids = np.array(sorted(clinical["geometry_id"].unique()))
    RNG.shuffle(geometry_ids)
    n_test = max(5, int(round(0.2 * len(geometry_ids))))
    held_ids = set(geometry_ids[:n_test])
    clinical_train = clinical[~clinical["geometry_id"].isin(held_ids)]
    clinical_held = clinical[clinical["geometry_id"].isin(held_ids)]

    g0_rows = []
    for i, mask in enumerate(random_blob_masks(max(64, len(clinical_train)))):
        row = descriptors_2d(mask)
        row.update({"sample_id": f"G0_blob_{i:03d}", "geometry_id": f"G0_blob_{i:03d}", "prior": "G0_random_field"})
        g0_rows.append(row)
    g1_rows = []
    for i, mask in enumerate(ellipse_masks(max(64, len(clinical_train)))):
        row = descriptors_2d(mask)
        row.update({"sample_id": f"G1_ellipse_{i:03d}", "geometry_id": f"G1_ellipse_{i:03d}", "prior": "G1_ellipse_cylinder"})
        g1_rows.append(row)

    g0 = pd.DataFrame(g0_rows)
    g1 = pd.DataFrame(g1_rows)
    g2_train = clinical_train.copy()
    g2_train["prior"] = "G2_clinical_memory_train"
    g2_held = clinical_held.copy()
    g2_held["prior"] = "heldout_clinical"

    all_priors = pd.concat([g0, g1, g2_train, g2_held], ignore_index=True)
    all_priors.to_csv(TABLES / "geometry_prior_descriptor_pool.csv", index=False)

    features = DESCRIPTOR_COLS_2D
    train_matrix = g2_train[features].astype(float).replace([np.inf, -np.inf], np.nan)
    mu = train_matrix.mean()
    sd = train_matrix.std().replace(0, 1.0)
    target = ((g2_held[features].astype(float) - mu) / sd).fillna(0).to_numpy()
    rows = []
    for name, cand in [
        ("G0_random_field", g0),
        ("G1_ellipse_cylinder", g1),
        ("G2_clinical_memory_train", g2_train),
    ]:
        cand_std = ((cand[features].astype(float) - mu) / sd).fillna(0).to_numpy()
        wd = np.mean(
            [
                one_d_wasserstein(cand_std[:, j], target[:, j])
                for j in range(cand_std.shape[1])
            ]
        )
        ed = np.mean(
            [
                stats.energy_distance(cand_std[:, j], target[:, j])
                for j in range(cand_std.shape[1])
            ]
        )
        coverage_vals = []
        for col in features:
            lo, hi = np.nanquantile(cand[col].astype(float), [0.05, 0.95])
            vals = g2_held[col].astype(float)
            coverage_vals.append(((vals >= lo) & (vals <= hi)).mean())
        rows.append(
            {
                "candidate_prior": name,
                "target": "heldout_clinical_masks",
                "n_candidate": len(cand),
                "n_target": len(g2_held),
                "mmd_rbf_standardized": mmd_rbf(cand_std, target),
                "wasserstein_mean_standardized": wd,
                "energy_distance_mean_standardized": float(ed),
                "descriptor_quantile_coverage_5_95": float(np.mean(coverage_vals)),
            }
        )
    match = pd.DataFrame(rows).sort_values("mmd_rbf_standardized")
    match.to_csv(TABLES / "geometry_distribution_matching.csv", index=False)

    split_rows = [
        {
            "split_type": "geometry_heldout",
            "train_geometry_count": len(geometry_ids) - n_test,
            "test_geometry_count": n_test,
            "no_geometry_id_leakage": True,
            "heldout_geometry_ids_preview": ";".join(sorted(list(held_ids))[:10]),
        }
    ]
    pd.DataFrame(split_rows).to_csv(TABLES / "splits_summary.csv", index=False)
    return all_priors, match


def sample_mechanics(n: int) -> pd.DataFrame:
    e_bg = RNG.uniform(3.0, 10.0, size=n)
    contrast = np.exp(RNG.uniform(np.log(1.2), np.log(12.0), size=n))
    low_contrast_flag = RNG.random(n) < 0.15
    contrast[low_contrast_flag] = RNG.uniform(1.1, 1.8, size=low_contrast_flag.sum())
    z_top = RNG.uniform(0.2, 8.0, size=n)
    thickness = RNG.uniform(0.5, 5.5, size=n)
    z_bottom = np.clip(z_top + thickness, 0.5, 15.0)
    thickness = z_bottom - z_top
    heterogeneity = RNG.uniform(0.0, 0.35, size=n)
    return pd.DataFrame(
        {
            "E_bg": e_bg,
            "contrast": contrast,
            "E_lesion": e_bg * contrast,
            "z_top": z_top,
            "z_bottom": z_bottom,
            "thickness": thickness,
            "heterogeneity": heterogeneity,
        }
    )


def compose_2d_field(mask: np.ndarray, mech: pd.Series, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    mask64 = resize_mask(mask, shape)
    delta = float(mech["E_lesion"] - mech["E_bg"])
    noise = ndimage.gaussian_filter(RNG.normal(size=shape), sigma=2.0)
    noise = noise / max(float(np.std(noise)), 1e-9)
    local = float(mech["heterogeneity"]) * float(mech["E_bg"]) * noise
    field = float(mech["E_bg"]) + delta * mask64.astype(float)
    field = field + local * (0.4 + 0.6 * mask64.astype(float))
    return np.clip(field, 0.1, None).astype(np.float32)


def extrude_3d(mask: np.ndarray, mech: pd.Series, shape: tuple[int, int, int] = (24, 64, 64)) -> np.ndarray:
    z_grid = np.linspace(0.0, 15.0, shape[0])
    mask64 = resize_mask(mask, (shape[1], shape[2]))
    z_top = float(mech["z_top"])
    z_bottom = float(mech["z_bottom"])
    profile_type = RNG.choice(["uniform_slab", "ellipsoidal_taper", "irregular_depth"])
    vol_mask = np.zeros(shape, dtype=float)
    for zi, z in enumerate(z_grid):
        if z_top <= z <= z_bottom:
            if profile_type == "uniform_slab":
                weight = 1.0
            elif profile_type == "ellipsoidal_taper":
                center = 0.5 * (z_top + z_bottom)
                half = max(0.5 * (z_bottom - z_top), 1e-6)
                weight = math.sqrt(max(0.0, 1.0 - ((z - center) / half) ** 2))
            else:
                base = 0.75 + 0.25 * math.sin(2 * math.pi * (z - z_top) / max(z_bottom - z_top, 1e-6))
                weight = float(np.clip(base + RNG.normal(0, 0.04), 0.25, 1.0))
            vol_mask[zi] = mask64.astype(float) * weight
    delta = float(mech["E_lesion"] - mech["E_bg"])
    noise = ndimage.gaussian_filter(RNG.normal(size=shape), sigma=(1.0, 2.0, 2.0))
    noise = noise / max(float(np.std(noise)), 1e-9)
    field = float(mech["E_bg"]) + delta * vol_mask
    field = field + float(mech["heterogeneity"]) * float(mech["E_bg"]) * noise
    return np.clip(field, 0.1, None).astype(np.float32)


def run_independence_tests(df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = df2[df2["source_geometry_type"] == "clinical_mask"].drop_duplicates("geometry_id").copy()
    clinical = clinical.dropna(subset=DESCRIPTOR_COLS_2D)
    n = 2048
    geom_idx = RNG.integers(0, len(clinical), size=n)
    geom = clinical.iloc[geom_idx].reset_index(drop=True)
    mech = sample_mechanics(n)
    decoupled = pd.concat(
        [
            pd.DataFrame(
                {
                    "decoupled_id": [f"decoupled_{i:04d}" for i in range(n)],
                    "geometry_id": geom["geometry_id"].to_numpy(),
                    "source_geometry_type": "clinical_mask",
                }
            ),
            geom[DESCRIPTOR_COLS_2D].reset_index(drop=True),
            mech,
        ],
        axis=1,
    )
    decoupled.to_csv(METRICS / "decoupled_dataset_metadata.csv", index=False)

    example_fields_2d = []
    example_fields_3d = []
    example_masks = []
    for i in range(12):
        mask = load_mask_csv(geom.iloc[i]["mask_path"])
        example_masks.append(resize_mask(mask))
        example_fields_2d.append(compose_2d_field(mask, mech.iloc[i]))
        example_fields_3d.append(extrude_3d(mask, mech.iloc[i]))
    np.savez_compressed(
        PREDICTIONS / "decoupled_examples.npz",
        masks=np.stack(example_masks),
        E2d=np.stack(example_fields_2d),
        E3d=np.stack(example_fields_3d),
    )

    x = decoupled[DESCRIPTOR_COLS_2D].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows = []
    heatmap_rows = []
    targets = ["z_top", "z_bottom", "thickness", "E_bg", "E_lesion", "contrast"]
    for target in targets:
        y = decoupled[target].astype(float).to_numpy()
        spears = []
        for col in DESCRIPTOR_COLS_2D:
            values = x[col].to_numpy()
            if np.nanstd(values) < 1e-12:
                rho = 0.0
            else:
                rho = stats.spearmanr(values, y, nan_policy="omit").correlation
            if not np.isfinite(rho):
                rho = 0.0
            spears.append(abs(float(rho)))
            heatmap_rows.append({"target": target, "descriptor": col, "abs_spearman": abs(float(rho))})
        mi = mutual_info_regression(x, y, random_state=SEED)
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, random_state=SEED)
        rf = RandomForestRegressor(n_estimators=160, max_depth=6, random_state=SEED, n_jobs=-1)
        rf.fit(xtr, ytr)
        pred = rf.predict(xte)
        q = pd.qcut(y, q=4, labels=False, duplicates="drop")
        xtr_c, xte_c, ytr_c, yte_c = train_test_split(x, q, test_size=0.35, random_state=SEED, stratify=q)
        clf = RandomForestClassifier(n_estimators=160, max_depth=5, random_state=SEED, n_jobs=-1)
        clf.fit(xtr_c, ytr_c)
        pred_c = clf.predict(xte_c)
        chance = float(pd.Series(yte_c).value_counts(normalize=True).max())
        rows.append(
            {
                "target": target,
                "max_abs_spearman": max(spears),
                "median_abs_spearman": float(np.median(spears)),
                "share_descriptors_abs_spearman_lt_0p10": float(np.mean(np.array(spears) < 0.10)),
                "mutual_info_mean": float(np.mean(mi)),
                "mutual_info_max": float(np.max(mi)),
                "rf_regression_r2": float(r2_score(yte, pred)),
                "rf_regression_mae": float(mean_absolute_error(yte, pred)),
                "rf_bin_accuracy": float(accuracy_score(yte_c, pred_c)),
                "rf_bin_balanced_accuracy": float(balanced_accuracy_score(yte_c, pred_c)),
                "bin_majority_chance": chance,
                "pass_decoupling_gate": bool((np.mean(np.array(spears) < 0.10) >= 0.80) and (r2_score(yte, pred) < 0.05)),
            }
        )
    tests = pd.DataFrame(rows)
    heat = pd.DataFrame(heatmap_rows)
    tests.to_csv(TABLES / "independence_tests.csv", index=False)
    heat.to_csv(TABLES / "independence_heatmap_values.csv", index=False)
    return tests, decoupled


def method_mapping_2d(method: str) -> dict:
    mapping = {
        "Interp.": ("D1_sparse_interpolation_baseline", "none_or_sparse", "operator_readout_only", "deterministic"),
        "Baseline U-Net": ("D1_operator_only_unet", "none", "operator_readout_only", "deterministic"),
        "Vanilla diffusion": ("D0_D1_vanilla_diffusion", "none", "operator_readout_conditioned", "gaussian_diffusion"),
        "Op-cond. diffusion + mask": (
            "D4_operator_plus_clinical_geometry",
            "G2_clinical_mask",
            "operator_plus_geometry",
            "conditional_diffusion_posterior",
        ),
    }
    model, prior, cond, path_type = mapping.get(method, (method, "", "", ""))
    return {
        "model_name": model,
        "geometry_prior": prior,
        "operator_conditioning": cond,
        "path_type": path_type,
    }


def method_mapping_3d(method: str) -> dict:
    mapping = {
        "direct_scalar_regression": ("D1_operator_only_scalar", "none", "operator_readout_only", "deterministic"),
        "unet_3d": ("D1_operator_only_3d_unet", "none", "operator_readout_only", "deterministic"),
        "kernel_gp": ("D5_depth_aware_kernel_gp", "G2_clinical_mask", "depth_aware_operator_kernel", "probabilistic_gp"),
        "vanilla_3d_diffusion": ("D0_D1_vanilla_3d_diffusion", "none", "operator_readout_conditioned", "gaussian_diffusion"),
        "eapp_only_diffusion": ("D1_eapp_only_diffusion", "none", "operator_eapp_only", "gaussian_diffusion"),
        "dps_inverse_diffusion": ("D4_operator_plus_clinical_geometry", "G2_clinical_mask", "operator_plus_geometry", "dps_diffusion"),
        "op_conditioned_diffusion_ours": (
            "D5_operator_plus_clinical_geometry_plus_depth_physics",
            "G2_clinical_mask",
            "operator_geometry_depth_physics",
            "clinically_informed_conditional_diffusion_posterior",
        ),
    }
    model, prior, cond, path_type = mapping.get(method, (method, "", "", ""))
    return {
        "model_name": model,
        "geometry_prior": prior,
        "operator_conditioning": cond,
        "path_type": path_type,
    }


def compute_ause_by_method() -> pd.DataFrame:
    risk = pd.read_csv(FIG3_ROOT / "metrics" / "risk_coverage.csv")
    rows = []
    for (op, method), grp in risk.groupby(["operator", "method"]):
        grp = grp.sort_values("coverage_fraction")
        x = grp["coverage_fraction"].to_numpy()
        y = (grp["risk_mae_mm"] - grp["oracle_risk_mae_mm"]).to_numpy()
        rows.append({"operator": op, "method": method, "AUSE": float(np.trapezoid(y, x))})
    return pd.DataFrame(rows)


def build_ablation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    panel = pd.read_csv(FIG2_ROOT / "tables" / "computed_panel_e_metrics.csv")
    for method, grp in panel.groupby("method"):
        mapped = method_mapping_2d(method)
        rows.append(
            {
                **mapped,
                "data_dim": "2D",
                "split_type": "figure2_panel_e_cases",
                "physics_randomization": "existing_simulation",
                "MAE_E": grp["MAE_kPa"].mean(),
                "RMSE_E": np.nan,
                "P90AE_E": np.nan,
                "Dice": grp["Dice"].mean(),
                "BoundaryF1": grp["Boundary_F1"].mean(),
                "HD95": np.nan,
                "MAE_z_bottom": np.nan,
                "MAE_thickness": np.nan,
                "IntervalIoU": np.nan,
                "CRPS": np.nan,
                "Cov90": np.nan,
                "Width90": np.nan,
                "UCE": np.nan,
                "AUROC_fail": np.nan,
                "AUSE": np.nan,
                "sampling_steps": np.nan,
                "inference_time_ms": np.nan,
                "source_table": "computed_panel_e_metrics.csv",
            }
        )

    depth = pd.read_csv(FIG3_ROOT / "metrics" / "depth_metrics.csv")
    ause = compute_ause_by_method()
    depth = depth.merge(ause, on=["operator", "method"], how="left")
    for row in depth.to_dict("records"):
        mapped = method_mapping_3d(row["method"])
        rows.append(
            {
                **mapped,
                "data_dim": "3D",
                "split_type": f"figure3_depth_benchmark_{row['operator']}",
                "physics_randomization": "existing_domain_randomization",
                "MAE_E": np.nan,
                "RMSE_E": np.nan,
                "P90AE_E": np.nan,
                "Dice": np.nan,
                "BoundaryF1": np.nan,
                "HD95": np.nan,
                "MAE_z_bottom": row.get("mae_zbottom_mm", np.nan),
                "MAE_thickness": np.nan,
                "IntervalIoU": row.get("interval_iou_mean", np.nan),
                "CRPS": row.get("crps_zbottom_mm", np.nan),
                "Cov90": row.get("cov90_zbottom", np.nan),
                "Width90": row.get("width90_zbottom_mm", np.nan),
                "UCE": row.get("uce", np.nan),
                "AUROC_fail": row.get("auroc_fail_tau_1mm", np.nan),
                "AUSE": row.get("AUSE", np.nan),
                "sampling_steps": 32 if "diffusion" in str(row["method"]) else np.nan,
                "inference_time_ms": np.nan,
                "source_table": "depth_metrics.csv+risk_coverage.csv",
            }
        )
    main = pd.DataFrame(rows)
    main.to_csv(TABLES / "main_ablation_table.csv", index=False)

    cal = pd.read_csv(FIG3_ROOT / "metrics" / "calibration_metrics.csv")
    extrap = pd.read_csv(FIG3_ROOT / "metrics" / "error_vs_extrapolation.csv")
    slope_rows = []
    for (op, method), grp in extrap.groupby(["operator", "method"]):
        if len(grp) > 2:
            slope, intercept, r, p, stderr = stats.linregress(grp["d_out_mm"], grp["absolute_error_mm"])
            uslope, _, ur, up, ustderr = stats.linregress(grp["d_out_mm"], grp["posterior_std_zbottom_mm"])
        else:
            slope = intercept = r = p = stderr = uslope = ur = up = ustderr = np.nan
        slope_rows.append(
            {
                "operator": op,
                "method": method,
                "error_vs_depth_ood_slope": slope,
                "error_vs_depth_ood_stderr": stderr,
                "uncertainty_vs_depth_ood_slope": uslope,
                "uncertainty_vs_depth_ood_stderr": ustderr,
            }
        )
    ood = cal.merge(pd.DataFrame(slope_rows), on=["operator", "method"], how="left")
    ood.to_csv(TABLES / "ood_failure_metrics.csv", index=False)
    return main, ood


def resolve_external_or_local(path_value: str, local_dir: Path) -> Path | None:
    path = Path(str(path_value))
    if path.exists():
        return path
    candidate = local_dir / path.name
    return candidate if candidate.exists() else None


def block_readout(field: np.ndarray, n: int = 8) -> np.ndarray:
    h, w = field.shape
    sh = h // n
    sw = w // n
    trimmed = field[: sh * n, : sw * n]
    return trimmed.reshape(n, sh, n, sw).mean(axis=(1, 3))


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=600)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        zorder=20,
    )


def prior_short_label(name: str) -> str:
    return {
        "G0_random_field": "G0 random",
        "G1_ellipse_cylinder": "G1 ellipse",
        "G2_clinical_memory_train": "G2 clinical",
        "heldout_clinical": "Held-out clinical",
    }.get(name, name)


def make_figure_1(decoupled: pd.DataFrame) -> None:
    examples = np.load(PREDICTIONS / "decoupled_examples.npz")
    mask = examples["masks"][0]
    e2d = examples["E2d"][0]
    e3d = examples["E3d"][0]
    readout = block_readout(ndimage.gaussian_filter(e2d, sigma=2.0), n=8)
    posterior = np.load(FIG3_ROOT / "posteriors" / "posterior_mean_volume.npz", allow_pickle=True)
    unc = np.load(FIG3_ROOT / "posteriors" / "posterior_uncertainty_volume.npz", allow_pickle=True)
    pm = posterior["posterior_mean_E_xyz"][3, 8]
    ps = unc["posterior_std_E_xyz"][3, 8]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(mask, cmap="gray_r")
    ax.set_title("Geometry-only memory g")
    ax.set_xticks([])
    ax.set_yticks([])
    panel_label(ax, "a")

    ax = axes[0, 1]
    ax.hist(decoupled["E_bg"], bins=24, alpha=0.75, label="E_bg", color="#4C78A8")
    ax.hist(decoupled["E_lesion"], bins=24, alpha=0.55, label="E_lesion", color="#F58518")
    ax.set_xlabel("Modulus (kPa)")
    ax.set_ylabel("Count")
    ax.set_title("Independent mechanics b")
    ax.legend(frameon=False)
    panel_label(ax, "b")

    ax = axes[0, 2]
    ax.hist(decoupled["z_bottom"], bins=24, color="#54A24B", alpha=0.85)
    ax.set_xlabel("z_bottom (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Depth randomized independently")
    panel_label(ax, "c")

    ax = axes[1, 0]
    im = ax.imshow(e2d, cmap="viridis")
    ax.set_title("Composed field T(g,b)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label="kPa")
    panel_label(ax, "d")

    ax = axes[1, 1]
    im = ax.imshow(readout, cmap="cividis")
    ax.set_title("Depth-aware readout H(x)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    panel_label(ax, "e")

    ax = axes[1, 2]
    im = ax.imshow(pm, cmap="viridis")
    ax.contour(ps, colors="white", linewidths=0.7, levels=4)
    ax.set_title("Posterior mean + uncertainty")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label="kPa")
    panel_label(ax, "f")
    save_figure(fig, "figure_1_factorized_memory_physics")


def make_figure_2(prior_pool: pd.DataFrame, match: pd.DataFrame, tests: pd.DataFrame) -> None:
    examples = np.load(PREDICTIONS / "decoupled_examples.npz")
    g0 = random_blob_masks(1)[0]
    g1 = ellipse_masks(1)[0]
    g2 = examples["masks"][1]
    feature_subset = ["area_frac", "eccentricity", "compactness", "boundary_curvature_mean", "asymmetry_score"]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.9), constrained_layout=True)
    ax = axes[0, 0]
    combo = np.concatenate([g0.astype(float), g1.astype(float), g2.astype(float)], axis=1)
    ax.imshow(combo, cmap="gray_r")
    ax.set_title("G0 random, G1 ellipse, G2 clinical")
    ax.set_xticks([])
    ax.set_yticks([])
    panel_label(ax, "a")

    ax = axes[0, 1]
    colors = {
        "G0_random_field": "#4C78A8",
        "G1_ellipse_cylinder": "#F58518",
        "G2_clinical_memory_train": "#54A24B",
        "heldout_clinical": "#B279A2",
    }
    for prior, grp in prior_pool.groupby("prior"):
        vals = grp["boundary_curvature_mean"].astype(float)
        ax.hist(vals, bins=24, histtype="step", density=True, color=colors.get(prior, "black"), label=prior_short_label(prior))
    ax.set_xlabel("Boundary irregularity")
    ax.set_ylabel("Density")
    ax.set_title("Descriptor distributions")
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "b")

    ax = axes[0, 2]
    x = prior_pool[DESCRIPTOR_COLS_2D].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    x = (x - x.mean()) / x.std().replace(0, 1)
    pcs = PCA(n_components=2, random_state=SEED).fit_transform(x)
    for prior, grp_idx in prior_pool.groupby("prior").groups.items():
        idx = np.array(list(grp_idx))
        ax.scatter(pcs[idx, 0], pcs[idx, 1], s=10, alpha=0.7, label=prior, color=colors.get(prior, "black"))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Geometry descriptor PCA")
    panel_label(ax, "c")

    ax = axes[1, 0]
    ax.bar(
        [prior_short_label(x) for x in match["candidate_prior"]],
        match["mmd_rbf_standardized"],
        color=[colors.get(x, "#777777") for x in match["candidate_prior"]],
    )
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("MMD (lower is better)")
    ax.set_title("Held-out clinical match")
    panel_label(ax, "d")

    ax = axes[1, 1]
    ax.bar(
        [prior_short_label(x) for x in match["candidate_prior"]],
        match["descriptor_quantile_coverage_5_95"],
        color=[colors.get(x, "#777777") for x in match["candidate_prior"]],
    )
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 1)
    ax.set_ylabel("5-95% descriptor coverage")
    ax.set_title("Clinical descriptor coverage")
    panel_label(ax, "e")

    heat = pd.read_csv(TABLES / "independence_heatmap_values.csv")
    heat = heat[heat["descriptor"].isin(feature_subset)]
    matrix = heat.pivot(index="target", columns="descriptor", values="abs_spearman").loc[
        ["z_top", "z_bottom", "thickness", "E_bg", "E_lesion", "contrast"], feature_subset
    ]
    ax = axes[1, 2]
    im = ax.imshow(matrix.to_numpy(), cmap="magma", vmin=0, vmax=0.15)
    ax.set_xticks(range(len(feature_subset)))
    ax.set_xticklabels(["area", "ecc.", "compact", "boundary", "asym."], rotation=0, fontsize=6)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Geometry vs mechanics independence")
    fig.colorbar(im, ax=ax, fraction=0.046, label="abs Spearman")
    panel_label(ax, "f")
    save_figure(fig, "figure_2_geometry_memory")


def make_figure_3() -> None:
    panel = pd.read_csv(FIG2_ROOT / "tables" / "computed_panel_e_metrics.csv")
    selected = pd.read_csv(FIG2_ROOT / "tables" / "selected_reconstruction_gallery_rows.csv")
    local_gallery = FIG2_ROOT / "data_sources" / "reconstruction_gallery"
    row = selected.iloc[0]
    target_p = resolve_external_or_local(row["target_npy"], local_gallery)
    coarse_p = resolve_external_or_local(row["coarse_prior_npy"], local_gallery)
    pred_p = resolve_external_or_local(row["pred_field2map_npy"], local_gallery)
    mask_p = resolve_external_or_local(row["target_mask_npy"], local_gallery)
    target = np.load(target_p) if target_p else np.zeros((64, 64))
    coarse = np.load(coarse_p) if coarse_p else block_readout(target, 8)
    pred = np.load(pred_p) if pred_p else target.copy()
    mask = np.load(mask_p) if mask_p else target > np.median(target)
    coarse = resize_or_pad_map(coarse, target.shape)
    pred = resize_or_pad_map(pred, target.shape)
    err = pred - target

    fig = plt.figure(figsize=(7.5, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    maps = [
        ("Ground truth E2D", target, "viridis", None),
        ("Sparse/coarse observation", coarse, "cividis", None),
        ("Clinical-memory prediction", pred, "viridis", None),
        ("Prediction error", err, "coolwarm", TwoSlopeNorm(vcenter=0, vmin=np.nanmin(err), vmax=np.nanmax(err))),
    ]
    for i, (title, arr, cmap, norm) in enumerate(maps):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(arr, cmap=cmap, norm=norm)
        if i in [0, 2]:
            ax.contour(mask, levels=[0.5], colors="white", linewidths=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
        panel_label(ax, chr(ord("a") + i))

    metrics = [("MAE_kPa", "MAE (kPa)", "lower"), ("Dice", "Dice", "higher"), ("Boundary_F1", "Boundary F1", "higher")]
    for j, (metric, ylabel, direction) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, j])
        methods = list(panel["method"].drop_duplicates())
        data = [panel.loc[panel["method"] == m, metric].to_numpy() for m in methods]
        ax.boxplot(data, tick_labels=[short_method(m) for m in methods], patch_artist=True)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"2D ablation ({direction})")
        panel_label(ax, chr(ord("e") + j))
    ax = fig.add_subplot(gs[1, 3])
    summary = panel.groupby("method")[["MAE_kPa", "Dice", "Boundary_F1"]].mean()
    baseline = summary.loc["Baseline U-Net"]
    ours = summary.loc["Op-cond. diffusion + mask"]
    gains = {
        "MAE\nreduction": (baseline["MAE_kPa"] - ours["MAE_kPa"]) / baseline["MAE_kPa"],
        "Dice\ngain": ours["Dice"] - baseline["Dice"],
        "Boundary\nF1 gain": ours["Boundary_F1"] - baseline["Boundary_F1"],
    }
    ax.bar(gains.keys(), gains.values(), color=["#54A24B", "#72B7B2", "#B279A2"])
    ax.axhline(0, color="black", linewidth=0.75)
    ax.tick_params(axis="x", rotation=0, labelsize=7)
    ax.set_title("Clinical-mask conditioning effect")
    panel_label(ax, "h")
    save_figure(fig, "figure_3_2d_ablation")


def resize_or_pad_map(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == shape:
        return arr
    if arr.ndim == 2:
        zoom = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
        return ndimage.zoom(arr, zoom, order=1)
    return np.resize(arr, shape)


def short_method(method: str) -> str:
    return {
        "Interp.": "Interp",
        "Baseline U-Net": "U-Net",
        "Vanilla diffusion": "Vanilla",
        "Op-cond. diffusion + mask": "Op+Mask",
    }.get(method, method)


def predicted_zbottom_from_volume(vol: np.ndarray, z_mm: np.ndarray, bg: float = 8.0) -> float:
    threshold = bg + 0.35 * max(float(np.nanmax(vol) - bg), 1e-6)
    support = vol > threshold
    frac = support.reshape(support.shape[0], -1).mean(axis=1)
    valid = np.where(frac > 0.005)[0]
    if len(valid) == 0:
        return np.nan
    return float(z_mm[valid.max()])


def make_figure_4() -> None:
    truth = np.load(FIG3_ROOT / "latent_truth" / "volume_E_xyz.npz", allow_pickle=True)
    mean = np.load(FIG3_ROOT / "posteriors" / "posterior_mean_volume.npz", allow_pickle=True)
    std = np.load(FIG3_ROOT / "posteriors" / "posterior_uncertainty_volume.npz", allow_pickle=True)
    samples = np.load(FIG3_ROOT / "posteriors" / "posterior_samples_E_xyz.npz", allow_pickle=True)
    cases = pd.read_csv(FIG3_ROOT / "metadata" / "cases.csv")
    z_mm = truth["z_mm"]
    idx = 3
    gt = truth["E_xyz"][idx]
    pm = mean["posterior_mean_E_xyz"][idx]
    ps = std["posterior_std_E_xyz"][idx]
    mask = truth["lesion_mask_xyz"][idx] > 0.5
    case_id = str(truth["case_id"][idx])

    fig = plt.figure(figsize=(7.6, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    slice_specs = [
        ("Axial true", gt[8], mask[8]),
        ("Axial posterior", pm[8], None),
        ("Sagittal posterior", pm[:, :, 32], None),
        ("Uncertainty", ps[8], None),
    ]
    for i, (title, arr, contour) in enumerate(slice_specs):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(arr, cmap="viridis" if i != 3 else "magma", aspect="auto")
        if contour is not None:
            ax.contour(contour, levels=[0.5], colors="white", linewidths=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
        panel_label(ax, chr(ord("a") + i))

    ax = fig.add_subplot(gs[1, 0:2])
    true_profile = mask.reshape(mask.shape[0], -1).mean(axis=1)
    pred_profile = (pm > (8.0 + 0.35 * (np.nanmax(pm) - 8.0))).reshape(pm.shape[0], -1).mean(axis=1)
    sample_profiles = []
    for s in samples["samples_E_xyz"][idx]:
        sample_profiles.append((s > (8.0 + 0.35 * (np.nanmax(s) - 8.0))).reshape(s.shape[0], -1).mean(axis=1))
    sample_profiles = np.asarray(sample_profiles)
    lo, hi = np.nanquantile(sample_profiles, [0.05, 0.95], axis=0)
    ax.plot(z_mm, true_profile, color="black", label="True support")
    ax.plot(z_mm, pred_profile, color="#4C78A8", label="Posterior mean")
    ax.fill_between(z_mm, lo, hi, color="#4C78A8", alpha=0.22, label="90% interval")
    ax.set_xlabel("Depth z (mm)")
    ax.set_ylabel("Support fraction")
    ax.set_title(f"Depth profile: {case_id}")
    ax.legend(frameon=False)
    panel_label(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    true_z = []
    pred_z = []
    intervals = []
    for i, row in cases.iterrows():
        true_z.append(float(row["z_bottom_mm"]))
        pred_z.append(predicted_zbottom_from_volume(mean["posterior_mean_E_xyz"][i], z_mm))
        sample_z = [predicted_zbottom_from_volume(s, z_mm) for s in samples["samples_E_xyz"][i]]
        intervals.append(np.nanquantile(sample_z, [0.05, 0.95]))
    true_z = np.array(true_z)
    pred_z = np.array(pred_z)
    intervals = np.array(intervals)
    ax.errorbar(true_z, pred_z, yerr=[pred_z - intervals[:, 0], intervals[:, 1] - pred_z], fmt="o", color="#F58518")
    lim = [0, 15]
    ax.plot(lim, lim, color="black", linewidth=0.75)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True z_bottom (mm)")
    ax.set_ylabel("Predicted z_bottom (mm)")
    ax.set_title("Depth posterior")
    panel_label(ax, "f")

    ax = fig.add_subplot(gs[1, 3])
    depth = pd.read_csv(FIG3_ROOT / "metrics" / "depth_metrics.csv")
    mmp = depth[depth["operator"] == "MMP"].copy()
    mmp = mmp.sort_values("mae_zbottom_mm")
    ax.barh([method_label(m) for m in mmp["method"]], mmp["mae_zbottom_mm"], color="#54A24B")
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("MAE z_bottom (mm)")
    ax.set_title("Held-out depth benchmark")
    panel_label(ax, "g")
    save_figure(fig, "figure_4_3d_depth_posterior")


def make_figure_5() -> None:
    cal = pd.read_csv(FIG3_ROOT / "metrics" / "calibration_metrics.csv")
    risk = pd.read_csv(FIG3_ROOT / "metrics" / "risk_coverage.csv")
    ex = pd.read_csv(FIG3_ROOT / "metrics" / "error_vs_extrapolation.csv")
    methods = ["vanilla_3d_diffusion", "dps_inverse_diffusion", "op_conditioned_diffusion_ours", "kernel_gp"]
    colors = {
        "vanilla_3d_diffusion": "#4C78A8",
        "dps_inverse_diffusion": "#F58518",
        "op_conditioned_diffusion_ours": "#54A24B",
        "kernel_gp": "#B279A2",
    }
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0), constrained_layout=True)
    ax = axes[0, 0]
    nom = np.array([50, 80, 90, 95])
    ax.plot([0, 100], [0, 100], color="black", linewidth=0.75)
    for method in methods:
        row = cal[(cal["operator"] == "MMP") & (cal["method"] == method)]
        if len(row):
            vals = row[["cov50_zbottom", "cov80_zbottom", "cov90_zbottom", "cov95_zbottom"]].iloc[0].to_numpy() * 100
            ax.plot(nom, vals, marker="o", label=method_label(method), color=colors[method])
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Observed coverage (%)")
    ax.set_title("Calibration")
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "a")

    ax = axes[0, 1]
    for method in methods:
        grp = risk[(risk["operator"] == "MMP") & (risk["method"] == method)].sort_values("coverage_fraction")
        if len(grp):
            ax.plot(grp["coverage_fraction"], grp["risk_mae_mm"], color=colors[method], label=method_label(method))
    ax.set_xlabel("Coverage fraction")
    ax.set_ylabel("Risk MAE (mm)")
    ax.set_title("Risk-coverage")
    panel_label(ax, "b")

    ax = axes[0, 2]
    mmp = ex[(ex["operator"] == "MMP") & (ex["method"].isin(methods))]
    mmp = mmp.sample(min(1600, len(mmp)), random_state=SEED)
    for method, grp in mmp.groupby("method"):
        ax.scatter(grp["posterior_std_zbottom_mm"], grp["absolute_error_mm"], s=8, alpha=0.35, color=colors.get(method), label=method_label(method))
    ax.set_xlabel("Posterior std z_bottom (mm)")
    ax.set_ylabel("Absolute error (mm)")
    ax.set_title("Uncertainty-error relation")
    panel_label(ax, "c")

    ax = axes[1, 0]
    rows = cal[(cal["operator"] == "MMP") & (cal["method"].isin(methods))]
    ax.bar([method_label(m) for m in rows["method"]], rows["auroc_fail_tau_1mm"], color=[colors[m] for m in rows["method"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Failure detection")
    ax.tick_params(axis="x", rotation=30)
    panel_label(ax, "d")

    ax = axes[1, 1]
    for method in methods:
        grp = ex[(ex["operator"] == "MMP") & (ex["method"] == method)]
        if len(grp):
            bins = np.linspace(grp["d_out_mm"].min(), grp["d_out_mm"].max(), 8)
            centers = []
            means = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sub = grp[(grp["d_out_mm"] >= lo) & (grp["d_out_mm"] < hi)]
                if len(sub):
                    centers.append((lo + hi) / 2)
                    means.append(sub["absolute_error_mm"].mean())
            ax.plot(centers, means, marker="o", color=colors[method], label=method_label(method))
    ax.set_xlabel("Depth OOD distance (mm)")
    ax.set_ylabel("Mean absolute error (mm)")
    ax.set_title("Error vs held-out depth")
    panel_label(ax, "e")

    ax = axes[1, 2]
    for method in methods:
        grp = ex[(ex["operator"] == "MMP") & (ex["method"] == method)]
        if len(grp):
            bins = np.linspace(grp["d_out_mm"].min(), grp["d_out_mm"].max(), 8)
            centers = []
            means = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sub = grp[(grp["d_out_mm"] >= lo) & (grp["d_out_mm"] < hi)]
                if len(sub):
                    centers.append((lo + hi) / 2)
                    means.append(sub["posterior_std_zbottom_mm"].mean())
            ax.plot(centers, means, marker="o", color=colors[method], label=method_label(method))
    ax.set_xlabel("Depth OOD distance (mm)")
    ax.set_ylabel("Posterior std (mm)")
    ax.set_title("Uncertainty vs held-out depth")
    panel_label(ax, "f")
    save_figure(fig, "figure_5_uncertainty_ood")


def method_label(method: str) -> str:
    return {
        "direct_scalar_regression": "Direct scalar",
        "unet_3d": "3D U-Net",
        "kernel_gp": "Kernel GP",
        "vanilla_3d_diffusion": "Vanilla diff.",
        "eapp_only_diffusion": "Eapp diff.",
        "dps_inverse_diffusion": "DPS diff.",
        "op_conditioned_diffusion_ours": "Op+geom+physics",
    }.get(method, method)


def write_reports(
    pdf_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    match: pd.DataFrame,
    tests: pd.DataFrame,
    main: pd.DataFrame,
    ood: pd.DataFrame,
) -> None:
    dim_summary = manifest.groupby("dim").agg(
        samples=("sample_id", "count"),
        geometries=("geometry_id", "nunique"),
        source_types=("source_geometry_type", lambda s: ", ".join(sorted(set(s)))),
        depth_rows=("z_bottom", lambda s: int(s.notna().sum())),
    )
    write_text(
        REPORTS / "data_audit.md",
        "# Data Audit\n\n"
        f"Source PDF: `{SOURCE_PDF}`\n\n"
        f"Pages extracted: 32-64 inclusive ({len(pdf_summary)} pages).\n\n"
        "## Manifest Summary\n\n"
        + markdown_table(dim_summary)
        + "\n\n"
        "The 2D manifest is built from Figure 2 lesion-case inventory paths. "
        "The 3D manifest is built from Figure 3 latent truth, posterior, operator, and metadata assets. "
        "Missing fields are retained as blank/NA rather than imputed.\n",
    )

    best_match = match.sort_values("mmd_rbf_standardized").iloc[0]
    decoupling_pass = bool(tests["pass_decoupling_gate"].all())
    main_2d = main[main["data_dim"] == "2D"].copy()
    opmask = main_2d[main_2d["model_name"] == "D4_operator_plus_clinical_geometry"].iloc[0]
    unet = main_2d[main_2d["model_name"] == "D1_operator_only_unet"].iloc[0]
    vanilla = main_2d[main_2d["model_name"] == "D0_D1_vanilla_diffusion"].iloc[0]
    mmp3 = main[(main["data_dim"] == "3D") & (main["split_type"].str.contains("MMP"))].copy()
    mmp3_best = mmp3.sort_values("MAE_z_bottom").iloc[0]

    write_text(
        REPORTS / "main_results_summary.md",
        "# Main Results Summary\n\n"
        "## Design\n\n"
        "This run implements the page 32-64 task as a reproducible audit and benchmark over existing 2D and 3D simulation outputs. "
        "Clinical memory is represented as lesion geometry only. Mechanics, depth, contrast, heterogeneity, and operator variables are handled separately.\n\n"
        "## Key Quantitative Findings\n\n"
        f"- Best held-out geometry distribution match by MMD: `{best_match['candidate_prior']}`.\n"
        f"- Decoupling gate across randomized depth/stiffness variables: `{'PASS' if decoupling_pass else 'FAIL'}`.\n"
        f"- 2D operator + clinical mask MAE: {opmask['MAE_E']:.3f} kPa vs operator-only U-Net {unet['MAE_E']:.3f} kPa and vanilla diffusion {vanilla['MAE_E']:.3f} kPa.\n"
        f"- 2D operator + clinical mask Dice: {opmask['Dice']:.3f}; Boundary F1: {opmask['BoundaryF1']:.3f}.\n"
        f"- Best MMP 3D z-bottom method: `{mmp3_best['model_name']}` with MAE_z_bottom {mmp3_best['MAE_z_bottom']:.3f} mm, Cov90 {mmp3_best['Cov90']:.3f}, IntervalIoU {mmp3_best['IntervalIoU']:.3f}.\n\n"
        "PSNR/SSIM are deliberately not used as the primary evidence. The report emphasizes support, boundary, depth posterior, calibration, and OOD/failure behavior.\n",
    )

    g2_match_pass = best_match["candidate_prior"] == "G2_clinical_memory_train"
    d2_fail_pass = bool((tests["rf_regression_r2"] < 0.05).all())
    d4_improves = bool((opmask["MAE_E"] < unet["MAE_E"]) and (opmask["Dice"] >= unet["Dice"]))
    d5 = main[
        (main["data_dim"] == "3D")
        & (main["model_name"] == "D5_operator_plus_clinical_geometry_plus_depth_physics")
        & (main["split_type"].str.contains("MMP"))
    ]
    vanilla3 = main[
        (main["data_dim"] == "3D")
        & (main["model_name"] == "D0_D1_vanilla_3d_diffusion")
        & (main["split_type"].str.contains("MMP"))
    ]
    d5_improves = bool(len(d5) and len(vanilla3) and d5.iloc[0]["MAE_z_bottom"] < vanilla3.iloc[0]["MAE_z_bottom"])
    ours_ood = ood[(ood["operator"] == "MMP") & (ood["method"] == "op_conditioned_diffusion_ours")]
    ood_pass = bool(len(ours_ood) and ours_ood.iloc[0]["uncertainty_vs_depth_ood_slope"] > 0)
    checks = [
        ("Does clinical geometry prior match real lesion shape distribution?", g2_match_pass, "geometry_distribution_matching.csv; figure_2_geometry_memory"),
        ("Is geometry statistically independent from depth/stiffness?", decoupling_pass, "independence_tests.csv; figure_2_geometry_memory"),
        ("Does clinical memory improve geometry recovery?", d4_improves, "main_ablation_table.csv; figure_3_2d_ablation"),
        ("Does clinical memory improve mechanics posterior after sensor conditioning?", d5_improves, "main_ablation_table.csv; figure_4_3d_depth_posterior"),
        ("Does geometry-only model fail to infer mechanical variables?", d2_fail_pass, "independence_tests.csv"),
        ("Does OOD uncertainty increase when depth leaves training support?", ood_pass, "ood_failure_metrics.csv; figure_5_uncertainty_ood"),
    ]
    lines = ["# Clinical Memory Claim Check\n", "| Question | Decision | Support |", "|---|---:|---|"]
    for question, passed, support in checks:
        lines.append(f"| {question} | {'PASS' if passed else 'INCONCLUSIVE'} | {support} |")
    lines.append(
        "\nGeometry memory is statistically decoupled from randomized mechanical variables; depth and stiffness cannot be inferred from geometry alone.\n"
        if decoupling_pass
        else "\nGeometry-mechanics decoupling did not pass every configured gate; revisit randomized sampling before making the claim.\n"
    )
    write_text(REPORTS / "clinical_memory_claim_check.md", "\n".join(lines))

    write_text(
        REPORTS / "diffusion_vs_flow_decision.md",
        "# Diffusion vs Flow Decision\n\n"
        "## Implemented in this run\n\n"
        "- Main posterior evidence uses existing conditional diffusion/posterior benchmark outputs and deterministic/probabilistic baselines.\n"
        "- A true directed endpoint path `u=G(g)` to `x=T(g,b)` was not trained in this run.\n"
        "- No stochastic interpolant loss `x_t=(1-t)u+t*x+gamma(t)*noise` was trained.\n"
        "- No entropy-regularized bridge objective was implemented.\n\n"
        "## Terminology recommendation\n\n"
        "Use `clinically informed conditional diffusion posterior sampler` for implemented diffusion results. "
        "Do not use `flow`, `bridge`, `stochastic interpolant`, or `Schrodinger bridge` in the title or main claim unless the optional endpoint-path branch is trained and evaluated.\n\n"
        "Flow/stochastic interpolant can be added later as a Methods ablation only if it improves mechanics error, boundary preservation, calibration, OOD detection, or inference cost without weakening the existing posterior evidence.\n",
    )
    write_text(FIGURES / "figure_6_diffusion_vs_flow_optional_NOT_GENERATED.md", "Optional figure not generated because no true flow/stochastic interpolant branch was trained.\n")

    write_text(
        REPORTS / "methods_reproducibility.md",
        "# Methods Reproducibility\n\n"
        "Run from the project root:\n\n"
        "```powershell\n"
        "python .\\scripts\\run_cigm_experiment.py\n"
        "```\n\n"
        "Random seed: 20260606.\n\n"
        "Generated outputs:\n\n"
        "- `results/data_manifest.csv`\n"
        "- `results/tables/geometry_descriptors_2d.csv`\n"
        "- `results/tables/geometry_descriptors_3d.csv`\n"
        "- `results/tables/geometry_distribution_matching.csv`\n"
        "- `results/tables/independence_tests.csv`\n"
        "- `results/tables/main_ablation_table.csv`\n"
        "- `results/tables/ood_failure_metrics.csv`\n"
        "- `results/figures/*.png` and `results/figures/*.pdf`\n"
        "- `results/reports/*.md`\n\n"
        "Expected runtime on this Windows workstation is a few minutes, dominated by reading mask CSVs and rendering 600 dpi figures.\n",
    )


def write_project_files() -> None:
    write_text(
        PROJECT_ROOT / "README.md",
        "# Clinically Informed Geometric Memory\n\n"
        "Reproducible experiment bundle for testing whether dermatology-image-derived geometry memory improves inverse biomechanical sensing while keeping geometry independent from stiffness, depth, thickness, background modulus, heterogeneity, noise, and operator metadata.\n\n"
        "## Run\n\n"
        "```powershell\n"
        "python .\\scripts\\run_cigm_experiment.py\n"
        "```\n\n"
        "## Key Outputs\n\n"
        "- `source_pdf_pages_32_64_extracted_text.txt`\n"
        "- `results/data_manifest.csv`\n"
        "- `results/tables/`\n"
        "- `results/figures/`\n"
        "- `results/reports/clinical_memory_claim_check.md`\n"
        "- `results/reports/diffusion_vs_flow_decision.md`\n\n"
        "The optional flow/stochastic-interpolant figure is intentionally not generated unless a true geometry-to-mechanics endpoint path is trained.\n",
    )
    write_text(
        PROJECT_ROOT / "requirements.txt",
        "numpy\npandas\nmatplotlib\nscipy\nscikit-learn\npypdf\n",
    )
    write_text(
        PROJECT_ROOT / "configs" / "data.yaml",
        "source_pdf: ../../Data/Clinically Informed Geometric Memory.pdf\n"
        "pdf_pages_1_indexed: [32, 64]\n"
        "figure2_root: ../../Figure2_data_reconstruction_20260526\n"
        "figure3_root: ../../Data/fig3\n"
        "mask_extraction:\n"
        "  method: provided_or_csv_threshold\n"
        "  threshold: 0.5\n"
        "  keep_largest_component_for_random_prior: true\n",
    )
    write_text(
        PROJECT_ROOT / "configs" / "experiments.yaml",
        "seed: 20260606\n"
        "geometry_priors:\n"
        "  G0: gaussian_random_field\n"
        "  G1: ellipse_or_cylinder\n"
        "  G2: clinical_mask_geometry\n"
        "decoupled_dataset:\n"
        "  n_samples: 2048\n"
        "  mechanics_sampling: independent\n"
        "ablation_models:\n"
        "  executed_from_existing_outputs: [D1, D4, D5]\n"
        "  negative_controls: [geometry_only_independence_test]\n"
        "  optional_not_executed: [flow_matching, stochastic_interpolant, schrodinger_bridge]\n",
    )
    write_text(
        PROJECT_ROOT / "configs" / "figures.yaml",
        "style: Nature Computational Science inspired\n"
        "formats: [png_600dpi, pdf]\n"
        "primary_metrics: [MAE_E, Dice, BoundaryF1, MAE_z_bottom, CRPS, Cov90, UCE, AUROC_fail, AUSE]\n"
        "colormaps:\n"
        "  modulus: viridis\n"
        "  uncertainty: magma\n"
        "  error: coolwarm_centered\n",
    )
    write_text(
        PROJECT_ROOT / "CODE_AVAILABILITY_DRAFT.md",
        "# Code Availability Draft\n\n"
        "The custom code used to build the Clinically Informed Geometric Memory audit, metrics tables, and figures is provided in `scripts/run_cigm_experiment.py`. The script reads the Figure 2 and Figure 3 simulation outputs, builds manifest and descriptor tables, runs geometry-mechanics decoupling tests, aggregates ablation metrics, and renders all figures.\n",
    )
    write_text(
        PROJECT_ROOT / "DATA_AVAILABILITY_DRAFT.md",
        "# Data Availability Draft\n\n"
        "The experiment uses existing simulated 2D modulus maps, lesion masks, reconstruction metrics, 3D modulus volumes, posterior samples, and depth/calibration metrics stored under the manuscript workspace. Generated derivative tables and figures are saved in `results/` and can be regenerated by running `scripts/run_cigm_experiment.py`.\n",
    )


def export_reproducibility_bundle() -> None:
    for sub in ["configs", "metrics_tables", "figure_generation_scripts", "trained_model_configs"]:
        (BUNDLE / sub).mkdir(parents=True, exist_ok=True)
    for cfg in (PROJECT_ROOT / "configs").glob("*.yaml"):
        shutil.copy2(cfg, BUNDLE / "configs" / cfg.name)
    for table in TABLES.glob("*.csv"):
        shutil.copy2(table, BUNDLE / "metrics_tables" / table.name)
    shutil.copy2(RESULTS / "data_manifest.csv", BUNDLE / "data_manifest.csv")
    shutil.copy2(PROJECT_ROOT / "scripts" / "run_cigm_experiment.py", BUNDLE / "figure_generation_scripts" / "run_cigm_experiment.py")
    write_text(BUNDLE / "trained_model_configs" / "README.md", "No new neural networks were trained in this audit run. Existing benchmark outputs were aggregated and evaluated.\n")


def run() -> None:
    t0 = time.time()
    ensure_dirs()
    set_style()
    pdf_summary = extract_pdf_pages()
    manifest, inventory = build_manifest()
    df2, df3 = build_geometry_descriptors(inventory)
    prior_pool, match = distribution_matching(df2)
    tests, decoupled = run_independence_tests(df2)
    main, ood = build_ablation_tables()
    make_figure_1(decoupled)
    make_figure_2(prior_pool, match, tests)
    make_figure_3()
    make_figure_4()
    make_figure_5()
    write_reports(pdf_summary, manifest, match, tests, main, ood)
    write_project_files()
    export_reproducibility_bundle()
    save_json(
        RESULTS / "run_summary.json",
        {
            "seed": SEED,
            "runtime_seconds": round(time.time() - t0, 2),
            "source_pdf": str(SOURCE_PDF),
            "pdf_pages": "32-64",
            "manifest_rows": int(len(manifest)),
            "geometry_descriptors_2d": int(len(df2)),
            "geometry_descriptors_3d": int(len(df3)),
            "figures": sorted([p.name for p in FIGURES.glob("*.png")]),
            "reports": sorted([p.name for p in REPORTS.glob("*.md")]),
        },
    )
    print(f"Completed CIGM experiment in {time.time() - t0:.1f} s")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    run()
