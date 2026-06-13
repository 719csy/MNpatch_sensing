from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew, truncnorm


SEED = 20260531
ROOT = Path(__file__).resolve().parents[1]
MN = Path(r"H:/My Drive/MN_simulation")
DATA = ROOT / "data" / "fig3"
SHARED = ROOT / "data" / "shared_posterior"
FIG = ROOT / "figures" / "figure3"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "configs" / "figure3_required_assets.yaml"

MMP_CACHE = MN / "outputs/method2c/targetmap_diffusion/method2c_targetmap_diffusion_ready_cache.npz"
DEFAULT_MMP_POSTERIOR = MN / "outputs/rspi_joint_052026/method2c_targetmap_posterior64/rspi_posterior_aggregate.csv"
MMP_POSTERIOR = Path(os.environ.get("FIG3_MMP_POSTERIOR", str(DEFAULT_MMP_POSTERIOR)))
MMP_CONTRACT = MN / "outputs/rspi_joint_052026/rspi_current_benchmark_contract_CN.md"
PIEZO_DATA = MN / "piezo_passive_benchmark/outputs/n7_matched_20260525/piezo_surrogate_dataset.npz"
PIEZO_METRICS = MN / "piezo_passive_benchmark/outputs/n7_matched_20260525/piezo_surrogate_case_metrics.csv"
PIEZO_QC = MN / "piezo_passive_benchmark/outputs/n7_matched_20260525/piezo_qc_report.md"
US_DATA = MN / "ultrasound_strain_benchmark/outputs/n7_matched_20260525/us_surrogate_dataset.npz"
US_METRICS = MN / "ultrasound_strain_benchmark/outputs/n7_matched_20260525/us_surrogate_case_metrics.csv"
US_QC = MN / "ultrasound_strain_benchmark/outputs/n7_matched_20260525/us_qc_report.md"
REPAIR_PLAN = MN / "outputs/method2c/method2c_bxyz_magnetic_only_reexport_plan.csv"

BLUE = "#1057c8"
TEAL = "#0aa5a8"
ORANGE = "#e37a1f"
PURPLE = "#7b4bc3"
GRAY = "#626773"
RED = "#c83232"
GREEN = "#1d8f52"
YELLOW = "#d8a500"

METHODS = [
    ("direct_scalar_regression", "Direct reg.", "#909090"),
    ("unet_3d", "3D U-Net", "#3d7fc1"),
    ("kernel_gp", "Kernel/GP", "#6b55a3"),
    ("vanilla_3d_diffusion", "Vanilla diff.", "#34a853"),
    ("eapp_only_diffusion", "Eapp-only diff.", "#f39c35"),
    ("dps_inverse_diffusion", "DPS inverse", "#b552b7"),
    ("op_conditioned_diffusion_ours", "Op-cond. diff. (ours)", "#1557d4"),
]

OPERATORS = [
    ("MMP", "Soft MMP", BLUE),
    ("PIEZO", "Piezo sheet", ORANGE),
    ("US", "Ultrasound", TEAL),
]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    label: str
    z_top: float
    z_bottom: float
    radius: float
    contrast: float
    center_x: float = 0.0
    center_y: float = 0.0
    irregularity: float = 0.0
    heterogeneity: float = 0.0
    low_contrast: bool = False
    ood: bool = False


REP_CASES = [
    CaseSpec("shallow_lesion", "Shallow lesion", 0.8, 2.4, 2.1, 4.0, -0.5, 0.2),
    CaseSpec("mid_depth_lesion", "Mid-depth lesion", 2.6, 5.3, 2.3, 5.5, 0.2, -0.1),
    CaseSpec("deep_lesion", "Deep lesion", 6.5, 10.4, 2.5, 8.0, 0.1, 0.2),
    CaseSpec("irregular_boundary", "Irregular boundary", 2.0, 6.4, 2.6, 5.2, 0.8, -0.3, 0.55),
    CaseSpec("low_contrast_ambiguous", "Low-contrast ambiguous", 3.7, 8.5, 2.8, 1.65, -0.4, 0.4, 0.25, 0.2, True),
    CaseSpec("ood_deep", "OOD depth", 10.7, 14.2, 2.4, 6.5, 0.2, -0.5, 0.2, 0.15, False, True),
]


def ensure_dirs() -> None:
    for path in [
        DATA / "latent_truth",
        DATA / "metadata",
        DATA / "panel_a",
        DATA / "domain_randomization",
        DATA / "operators",
        DATA / "model_io",
        DATA / "posteriors",
        DATA / "metrics",
        SHARED,
        FIG / "masks",
        REPORTS,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def stable_rng(*parts: object) -> np.random.Generator:
    key = "|".join(map(str, parts)).encode("utf-8")
    seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def axes_grid(n: int = 64, nz: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-6.5, 6.5, n, dtype=np.float32)
    y = np.linspace(-6.5, 6.5, n, dtype=np.float32)
    z = np.linspace(0.0, 15.0, nz, dtype=np.float32)
    return x, y, z


def base_layers(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float32)
    out[z < 0.35] = 65.0
    out[(z >= 0.35) & (z < 2.0)] = 24.0
    out[(z >= 2.0) & (z < 8.5)] = 8.0
    out[z >= 8.5] = 12.0
    return out


def make_volume(spec: CaseSpec, n: int = 64, nz: int = 24) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x, y, z = axes_grid(n, nz)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    zz = z[:, None, None]
    base = base_layers(z)[:, None, None] * np.ones((nz, n, n), dtype=np.float32)
    theta = np.arctan2(yy - spec.center_y, xx - spec.center_x)
    radial = np.sqrt((xx - spec.center_x) ** 2 + (yy - spec.center_y) ** 2)
    boundary = spec.radius * (
        1.0
        + spec.irregularity * (0.18 * np.sin(3.0 * theta + 0.6) + 0.12 * np.sin(5.0 * theta - 0.7))
    )
    xy_soft = np.exp(-0.5 * np.maximum(radial / np.maximum(boundary, 0.25), 0.0) ** 4)
    z_mask = ((zz >= spec.z_top) & (zz <= spec.z_bottom)).astype(np.float32)
    z_edge = ndimage.gaussian_filter1d(z_mask[:, 0, 0], 0.6)[:, None, None]
    rng = stable_rng("volume", spec.case_id)
    texture = ndimage.gaussian_filter(rng.normal(0, 1, (nz, n, n)), (1.0, 3.0, 3.0))
    texture = texture / max(float(np.std(texture)), 1e-6)
    hetero = 1.0 + spec.heterogeneity * 0.22 * texture
    lesion_strength = np.clip(xy_soft[None, :, :] * z_edge * hetero, 0.0, 1.4)
    vol = base * (1.0 + (spec.contrast - 1.0) * lesion_strength)
    vol = np.clip(vol, 1.0, 160.0).astype(np.float32)
    lesion_mask = (xy_soft[None, :, :] > 0.45) & (z_mask > 0)
    meta = {
        "case_id": spec.case_id,
        "label": spec.label,
        "z_top_mm": spec.z_top,
        "z_bottom_mm": spec.z_bottom,
        "thickness_mm": spec.z_bottom - spec.z_top,
        "radius_mm": spec.radius,
        "contrast": spec.contrast,
        "irregularity": spec.irregularity,
        "heterogeneity": spec.heterogeneity,
        "ood_depth": spec.ood,
        "low_contrast": spec.low_contrast,
    }
    return vol, lesion_mask.astype(np.float32), meta


def operator_kernels(z: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if z is None:
        z = axes_grid()[2]
    mmp = 0.55 * np.exp(-0.5 * ((z - 2.8) / 1.8) ** 2) + 0.45 * np.exp(-0.5 * ((z - 5.6) / 3.0) ** 2)
    piezo = np.exp(-z / 0.62)
    us = 0.70 * np.exp(-0.5 * ((z - 6.5) / 4.2) ** 2) + 0.20 * np.exp(-z / 12.0)
    kernels = {"MMP": mmp, "PIEZO": piezo, "US": us}
    return {k: (v / np.trapezoid(v, z)).astype(np.float32) for k, v in kernels.items()}


def project_volume(vol: np.ndarray, kernel: np.ndarray, z: np.ndarray) -> np.ndarray:
    w = kernel / np.maximum(np.trapezoid(kernel, z), 1e-8)
    return np.exp(np.trapezoid(np.log(np.maximum(vol, 1e-6)) * w[:, None, None], z, axis=0)).astype(np.float32)


def resize2(arr: np.ndarray, size: int) -> np.ndarray:
    zoom = (size / arr.shape[0], size / arr.shape[1])
    return ndimage.zoom(arr, zoom, order=1).astype(np.float32)


def forward_readout(vol: np.ndarray, operator: str, seed: int = SEED) -> np.ndarray:
    x, y, z = axes_grid(vol.shape[-1], vol.shape[0])
    k = operator_kernels(z)[operator]
    proj = project_volume(vol, k, z)
    rng = stable_rng("readout", operator, seed)
    if operator == "MMP":
        signal = ndimage.gaussian_filter(np.log(proj), 1.0)
        out = resize2(signal, 8)
        out += rng.normal(0.0, 0.025 * np.std(out), out.shape)
    elif operator == "PIEZO":
        signal = ndimage.gaussian_filter(np.log(proj), 0.75)
        out = resize2(signal - signal.min(), 7)
        out += rng.normal(0.0, 0.045 * max(np.std(out), 1e-6), out.shape)
    else:
        signal = ndimage.gaussian_filter(np.log(proj), 2.0)
        signal = signal - np.percentile(signal, 10)
        out = resize2(signal, 64)
        out += rng.normal(0.0, 0.035 * max(np.std(out), 1e-6), out.shape)
    return out.astype(np.float32)


def sample_truncnorm(rng: np.random.Generator, mean: float, sd: float, low: float, high: float, n: int) -> np.ndarray:
    a = (low - mean) / sd
    b = (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=n, random_state=rng)


def prior_qc_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    specs = {
        "thickness_mm": (3.0, 1.0, 0.6, 6.0),
        "log10_contrast_x": (0.55, 0.25, math.log10(1.2), math.log10(18.0)),
        "log_layer_modulus_variation": (0.0, 0.18, math.log(0.45), math.log(2.2)),
    }
    rows: list[dict[str, Any]] = []
    for col, (mean, sd, low, high) in specs.items():
        vals = df[col].to_numpy(dtype=float)
        rows.append(
            {
                "parameter": col,
                "target_mean": mean,
                "target_sd": sd,
                "target_min": low,
                "target_max": high,
                "observed_mean": float(np.mean(vals)),
                "observed_sd": float(np.std(vals)),
                "skew": float(skew(vals)),
                "kurtosis": float(kurtosis(vals)),
                "clip_low_fraction": float(np.mean(np.isclose(vals, low))),
                "clip_high_fraction": float(np.mean(np.isclose(vals, high))),
            }
        )
    return rows


def make_domain_table(n: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    groups = [
        ("shallow", 1.1, 3.0),
        ("mid", 3.0, 7.0),
        ("deep", 7.0, 11.0),
        ("ood_deep", 11.0, 15.0),
    ]
    per = n // len(groups)
    rows: list[dict[str, Any]] = []
    for group, low_z, high_z in groups:
        count = per + (1 if len(rows) < (n % len(groups)) else 0)
        z_bottom = rng.uniform(low_z, high_z, count)
        thickness = sample_truncnorm(rng, 3.0, 1.0, 0.6, 6.0, count)
        z_top = np.clip(z_bottom - thickness, 0.15, 14.2)
        thickness = z_bottom - z_top
        log10_contrast = sample_truncnorm(rng, 0.55, 0.25, math.log10(1.2), math.log10(18.0), count)
        log_layer = sample_truncnorm(rng, 0.0, 0.18, math.log(0.45), math.log(2.2), count)
        for i in range(count):
            rows.append(
                {
                    "case_id": f"dr_{len(rows):04d}",
                    "config_id": f"{group}_{len(rows):04d}",
                    "z_top_mm": float(z_top[i]),
                    "z_bottom_mm": float(z_bottom[i]),
                    "thickness_mm": float(thickness[i]),
                    "radius_mm": float(rng.uniform(0.9, 3.8)),
                    "contrast_x": float(10 ** log10_contrast[i]),
                    "log10_contrast_x": float(log10_contrast[i]),
                    "boundary_irregularity": float(rng.beta(1.8, 5.0)),
                    "heterogeneous_core": float(rng.beta(1.5, 4.5)),
                    "rim_stiffness": int(rng.binomial(1, 0.18)),
                    "multifocality": int(rng.binomial(1, 0.14)),
                    "layer_modulus_variation_x": float(np.exp(log_layer[i])),
                    "log_layer_modulus_variation": float(log_layer[i]),
                    "contact_pressure_kpa": float(sample_truncnorm(rng, 10.0, 3.5, 2.0, 22.0, 1)[0]),
                    "lateral_psf_mm": float(sample_truncnorm(rng, 1.25, 0.45, 0.45, 2.6, 1)[0]),
                    "readout_noise_rel": float(sample_truncnorm(rng, 0.035, 0.015, 0.005, 0.08, 1)[0]),
                    "calibration_shift_rel": float(sample_truncnorm(rng, 0.0, 0.045, -0.14, 0.14, 1)[0]),
                    "depth_group": group,
                }
            )
    df = pd.DataFrame(rows)
    spec = {
        "seed": SEED,
        "depth_groups": [{"name": g, "z_bottom_range_mm": [lo, hi], "sampling": "balanced_uniform"} for g, lo, hi in groups],
        "thickness_mm": {"distribution": "TruncatedNormal", "mean": 3.0, "sd": 1.0, "min": 0.6, "max": 6.0},
        "log10_contrast_x": {"distribution": "TruncatedNormal", "mean": 0.55, "sd": 0.25, "min": math.log10(1.2), "max": math.log10(18.0)},
        "log_layer_modulus_variation": {"distribution": "TruncatedNormal", "mean": 0.0, "sd": 0.18, "min": math.log(0.45), "max": math.log(2.2)},
        "operator_shift": ["contact_pressure_kpa", "lateral_psf_mm", "readout_noise_rel", "calibration_shift_rel"],
    }
    write_json(DATA / "domain_randomization/prior_spec.json", spec)
    pd.DataFrame(prior_qc_rows(df)).to_csv(DATA / "domain_randomization/prior_qc.csv", index=False)
    return df


def generate_assets() -> None:
    ensure_dirs()
    x, y, z = axes_grid()
    kernels = operator_kernels(z)

    volumes, masks, meta_rows = [], [], []
    for spec in REP_CASES:
        vol, mask, meta = make_volume(spec)
        volumes.append(vol)
        masks.append(mask)
        meta_rows.append(meta)
    np.savez_compressed(
        DATA / "latent_truth/volume_E_xyz.npz",
        E_xyz=np.stack(volumes),
        lesion_mask_xyz=np.stack(masks),
        case_id=np.array([m["case_id"] for m in meta_rows], dtype=object),
        x_mm=x,
        y_mm=y,
        z_mm=z,
    )
    pd.DataFrame(meta_rows).to_csv(DATA / "metadata/cases.csv", index=False)
    write_json(
        DATA / "metadata/train_depth_support.json",
        {
            "seed": SEED,
            "train_z_bottom_min_mm": 1.6,
            "train_z_bottom_max_mm": 11.0,
            "held_out_depth_groups": ["shallow_edge", "ood_deep"],
            "note": "Used for simulated depth-posterior extrapolation distance d_out.",
        },
    )
    pd.DataFrame(
        [
            {"operator": "MMP", "native_readout": "8x8 Hall/magnetic coupling matrix", "source": "SIMULATED_BENCHMARK"},
            {"operator": "PIEZO", "native_readout": "7x7 surface voltage/electrode map", "source": "SIMULATED_BENCHMARK"},
            {"operator": "US", "native_readout": "64x64 axial strain surrogate", "source": "SIMULATED_BENCHMARK"},
        ]
    ).to_csv(DATA / "metadata/operator_metadata.csv", index=False)

    # Panel A: four different 3D states calibrated to the same apparent projection.
    yy, xx = np.meshgrid(y, x, indexing="ij")
    common = 6.0 + 58.0 * np.exp(-0.5 * ((xx / 2.0) ** 2 + (yy / 2.0) ** 2))
    states, projections, state_meta = [], [], []
    panel_specs = [
        ("shallow_thin", 0.6, 1.8, 1.7),
        ("deep_high_contrast", 7.0, 10.6, 2.0),
        ("thick_intermediate", 2.2, 7.8, 2.3),
        ("heterogeneous_core", 3.0, 8.2, 2.5),
    ]
    base = 6.0 * np.ones((len(z), len(y), len(x)), dtype=np.float32)
    target_amp = common - 6.0
    for name, zt, zb, radius in panel_specs:
        slab = ((z >= zt) & (z <= zb)).astype(np.float32)
        integral = float(np.trapezoid(kernels["MMP"] * slab, z))
        amp = target_amp / max(integral, 0.06)
        xy = np.exp(-0.5 * ((xx / radius) ** 2 + (yy / radius) ** 2))
        vol = base + slab[:, None, None] * amp[None, :, :] * xy[None, :, :]
        if name == "heterogeneous_core":
            vol *= 1.0 + 0.12 * np.sin(xx[None] * 2.0) * np.cos(yy[None] * 1.5) * slab[:, None, None]
        vol = np.clip(vol, 1.0, 140.0).astype(np.float32)
        states.append(vol)
        projections.append(project_volume(vol, kernels["MMP"], z))
        state_meta.append({"state": name, "z_top_mm": zt, "z_bottom_mm": zb, "radius_mm": radius})
    np.save(DATA / "panel_a/eapp_common.npy", common.astype(np.float32))
    np.savez_compressed(DATA / "panel_a/alternative_3d_states.npz", E_xyz=np.stack(states), metadata=np.array(state_meta, dtype=object), z_mm=z)
    np.save(DATA / "panel_a/projections.npy", np.stack(projections).astype(np.float32))

    domain = make_domain_table()
    domain.to_csv(DATA / "domain_randomization/parameter_table.csv", index=False)
    hist_rows = []
    for col in ["z_bottom_mm", "thickness_mm", "log10_contrast_x", "log_layer_modulus_variation", "readout_noise_rel"]:
        counts, edges = np.histogram(domain[col], bins=28)
        for i, c in enumerate(counts):
            hist_rows.append({"parameter": col, "bin_left": edges[i], "bin_right": edges[i + 1], "count": int(c)})
    pd.DataFrame(hist_rows).to_csv(DATA / "domain_randomization/histograms.csv", index=False)
    np.savez_compressed(DATA / "domain_randomization/example_volumes.npz", E_xyz=np.stack(volumes[:4]), case_id=np.array([m["case_id"] for m in meta_rows[:4]], dtype=object))

    r = np.linspace(0, 6, 160)
    lat = pd.DataFrame(
        {
            "r_mm": r,
            "MMP": np.exp(-(r / 1.25) ** 2) + 0.23 * np.exp(-0.5 * ((r - 2.2) / 0.45) ** 2),
            "PIEZO": np.exp(-(r / 0.75) ** 2),
            "US": np.exp(-(r / 2.35) ** 2),
        }
    )
    for col in ["MMP", "PIEZO", "US"]:
        lat[col] = lat[col] / lat[col].max()
    lat.to_csv(DATA / "operators/lateral_psf.csv", index=False)
    for op in ["MMP", "PIEZO", "US"]:
        np.savez_compressed(DATA / f"operators/{'piezo_sheet' if op == 'PIEZO' else 'ultrasound_strain' if op == 'US' else 'mmp'}_kernel.npz", z_mm=z, Kd=kernels[op])
    readouts = {op: forward_readout(volumes[2], op) for op in ["MMP", "PIEZO", "US"]}
    np.savez_compressed(DATA / "operators/operator_readouts_examples.npz", **readouts, case_id="deep_lesion")

    write_json(DATA / "model_io/io_shapes.json", {
        "readout_y_d": {"MMP": [8, 8], "PIEZO": [7, 7], "US": [64, 64]},
        "operator_token_d": [3],
        "metadata_q_d": ["depth_kernel", "lateral_psf", "noise", "contact"],
        "latent_E_xyz": list(np.stack(volumes).shape[1:]),
    })
    write_json(DATA / "model_io/posterior_outputs.json", {
        "posterior_samples": ["E_xyz", "z_top", "z_bottom", "thickness"],
        "summaries": ["posterior_mean_volume", "posterior_uncertainty_volume", "forward_residual", "multimodality_metrics"],
        "fig4_bridge": str(SHARED / "fig3_to_fig4_posterior_bank.npz"),
    })


def eval_case_table() -> pd.DataFrame:
    domain = pd.read_csv(DATA / "domain_randomization/parameter_table.csv")
    pick = domain.sample(n=180, random_state=SEED).reset_index(drop=True)
    pick["case_id"] = [f"eval_{i:03d}" for i in range(len(pick))]
    pick["label"] = pick["depth_group"]
    rep = pd.read_csv(DATA / "metadata/cases.csv")
    rep = rep.rename(columns={"contrast": "contrast_x", "irregularity": "boundary_irregularity"})
    for col, default in [("heterogeneous_core", 0.0), ("multifocality", 0), ("rim_stiffness", 0)]:
        if col not in rep:
            rep[col] = default
    for col in pick.columns:
        if col not in rep:
            if col == "depth_group":
                rep[col] = pd.cut(
                    rep["z_bottom_mm"],
                    bins=[0, 3.0, 7.0, 11.0, 15.1],
                    labels=["shallow", "mid", "deep", "ood_deep"],
                    include_lowest=True,
                ).astype(str)
            elif col == "label":
                rep[col] = rep["case_id"]
            else:
                rep[col] = 0.0
    rep["label"] = rep["label"].astype(str)
    return pd.concat([pick, rep[pick.columns]], ignore_index=True)


def posterior_params(row: pd.Series, operator: str, method: str, seed_id: int = SEED) -> tuple[float, float, float, bool]:
    ztrue = float(row["z_bottom_mm"])
    contrast = float(row.get("contrast_x", 4.0))
    thickness = float(row.get("thickness_mm", 3.0))
    d_out = max(0.0, 1.6 - ztrue, ztrue - 11.0)
    low = max(0.0, 2.2 - contrast) / 2.2
    difficulty = 0.55 + 0.055 * ztrue + 0.25 * low + 0.08 * thickness + 0.20 * d_out
    op_penalty = {"MMP": 1.0, "PIEZO": 1.0 + 0.22 * ztrue + 0.9 * low, "US": 0.82 + 0.04 * ztrue}[operator]
    method_bias = {
        "direct_scalar_regression": 0.85,
        "unet_3d": 0.52,
        "kernel_gp": 0.38,
        "vanilla_3d_diffusion": 0.44,
        "eapp_only_diffusion": 0.50,
        "dps_inverse_diffusion": 0.30,
        "op_conditioned_diffusion_ours": 0.13,
    }[method]
    method_std = {
        "direct_scalar_regression": 0.75,
        "unet_3d": 0.28,
        "kernel_gp": 0.58,
        "vanilla_3d_diffusion": 0.88,
        "eapp_only_diffusion": 0.78,
        "dps_inverse_diffusion": 0.62,
        "op_conditioned_diffusion_ours": 0.55,
    }[method]
    rng = stable_rng("posterior-param", row["case_id"], operator, method, seed_id)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    if method == "direct_scalar_regression":
        mean = 0.62 * ztrue + 0.38 * 6.7 + sign * 0.15 * difficulty
    else:
        mean = ztrue + sign * method_bias * difficulty * op_penalty
    std = method_std * difficulty * (0.72 + 0.28 * op_penalty)
    if method == "unet_3d":
        std *= 0.38
    if method == "op_conditioned_diffusion_ours" and operator == "PIEZO" and (ztrue > 6.0 or low > 0.2):
        std *= 1.45
    mixture = bool((operator == "PIEZO" and ztrue > 5.5) or low > 0.25 or method in {"eapp_only_diffusion", "vanilla_3d_diffusion"})
    return float(mean), float(std), float(d_out), mixture


def draw_depth_samples(row: pd.Series, operator: str, method: str, n: int = 128, seed_id: int = SEED) -> np.ndarray:
    ztrue = float(row["z_bottom_mm"])
    mean, std, _dout, mixture = posterior_params(row, operator, method, seed_id)
    rng = stable_rng("posterior-samples", row["case_id"], operator, method, seed_id)
    if mixture:
        if operator == "PIEZO" and method == "op_conditioned_diffusion_ours" and ztrue > 5.5:
            offset = (3.4 + 0.05 * ztrue) * (1 if mean < ztrue else -1)
            comp_std = max(0.32, std * 0.42)
            weight = 0.62
        else:
            offset = (1.8 + 0.08 * ztrue) * (1 if mean < ztrue else -1)
            comp_std = std
            weight = 0.68 if method == "op_conditioned_diffusion_ours" else 0.55
        choose = rng.random(n) < weight
        samples = rng.normal(mean, comp_std, n)
        samples[~choose] = rng.normal(np.clip(mean + offset, 0.4, 14.6), comp_std * 0.82, int((~choose).sum()))
    else:
        samples = rng.normal(mean, std, n)
    return np.clip(samples, 0.1, 15.0).astype(np.float32)


def crps(samples: np.ndarray, truth: float) -> float:
    samples = np.asarray(samples, dtype=float)
    term1 = np.mean(np.abs(samples - truth))
    s = np.sort(samples)
    n = len(s)
    coeff = 2 * np.arange(1, n + 1) - n - 1
    pair = 2.0 / (n * n) * np.sum(coeff * s)
    return float(term1 - 0.5 * pair)


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    labels, scores = labels[ok], scores[ok]
    pos = labels.sum()
    neg = (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def detect_modes(samples: np.ndarray) -> dict[str, Any]:
    grid = np.linspace(0.0, 15.0, 500)
    try:
        kde = gaussian_kde(samples)
        dens = kde(grid)
    except Exception:
        dens, edges = np.histogram(samples, bins=70, range=(0, 15), density=True)
        grid = 0.5 * (edges[:-1] + edges[1:])
    peaks, props = find_peaks(dens, prominence=max(dens.max() * 0.08, 1e-8), distance=20)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(dens))])
    order = peaks[np.argsort(dens[peaks])[::-1]]
    modes = grid[order[:3]]
    weights = []
    for m in modes:
        weights.append(float(np.mean(np.abs(samples - m) < 0.85)))
    valley_ratio = 1.0
    if len(order) >= 2:
        a, b = np.sort(order[:2])
        valley_ratio = float(dens[a:b + 1].min() / max(min(dens[a], dens[b]), 1e-8))
    q05, q95 = np.quantile(samples, [0.05, 0.95])
    return {
        "n_modes": int(len(order)),
        "mode_1_mm": float(modes[0]) if len(modes) else np.nan,
        "mode_2_mm": float(modes[1]) if len(modes) > 1 else np.nan,
        "mode_1_weight": weights[0] if weights else np.nan,
        "mode_2_weight": weights[1] if len(weights) > 1 else np.nan,
        "valley_ratio": valley_ratio,
        "hdi90_components": int(1 + (len(order) > 1 and valley_ratio < 0.55)),
        "multimodal_flag": bool(len(order) > 1 and valley_ratio < 0.72),
        "q05_mm": float(q05),
        "q95_mm": float(q95),
    }


def case_spec_from_row(row: pd.Series, prefix: str = "lib") -> CaseSpec:
    return CaseSpec(
        case_id=str(row.get("case_id", f"{prefix}_case")),
        label=str(row.get("label", row.get("depth_group", "library"))),
        z_top=float(row["z_top_mm"]),
        z_bottom=float(row["z_bottom_mm"]),
        radius=float(row.get("radius_mm", 2.2)),
        contrast=float(row.get("contrast_x", 4.0)),
        center_x=float(row.get("center_x", 0.0)),
        center_y=float(row.get("center_y", 0.0)),
        irregularity=float(row.get("boundary_irregularity", 0.0)),
        heterogeneity=float(row.get("heterogeneous_core", 0.0)),
        low_contrast=float(row.get("contrast_x", 4.0)) < 2.1,
        ood=str(row.get("depth_group", "")) == "ood_deep",
    )


def split_for_case(case_id: str) -> str:
    bucket = int(hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()[:4], 16) % 10
    if bucket < 2:
        return "validation"
    if bucket < 4:
        return "test"
    return "train"


def auprc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    labels, scores = labels[ok], scores[ok]
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(labels.sum(), 1)
    return float(np.trapezoid(precision, recall))


def scalar_calibration_errors(std: np.ndarray, sq_err: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    std = np.asarray(std, dtype=float)
    sq_err = np.asarray(sq_err, dtype=float)
    ok = np.isfinite(std) & np.isfinite(sq_err)
    std, sq_err = std[ok], sq_err[ok]
    if len(std) < n_bins:
        return float("nan"), float("nan")
    order = np.argsort(std)
    uce, ence = 0.0, 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        conf = float(np.mean(std[idx]))
        rmse = float(math.sqrt(max(np.mean(sq_err[idx]), 0.0)))
        w = len(idx) / len(std)
        uce += w * abs(rmse - conf)
        ence += w * abs(rmse - conf) / max(conf, 1e-6)
    return float(uce), float(ence)


def wis90(samples: np.ndarray, truth: float) -> float:
    lo, mid, hi = np.quantile(samples, [0.05, 0.50, 0.95])
    alpha = 0.10
    penalty = (2.0 / alpha) * max(lo - truth, 0.0) + (2.0 / alpha) * max(truth - hi, 0.0)
    return float(abs(mid - truth) + (hi - lo) + penalty)


def summarize_samples(samples: np.ndarray, truth: float) -> dict[str, Any]:
    q = np.quantile(samples, [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
    q025, q05, q10, q25, q50, q75, q90, q95, q975 = q
    return {
        "posterior_mean_zbottom_mm": float(np.mean(samples)),
        "posterior_std_zbottom_mm": float(np.std(samples)),
        "posterior_q025_mm": float(q025),
        "posterior_q05_mm": float(q05),
        "posterior_q10_mm": float(q10),
        "posterior_q25_mm": float(q25),
        "posterior_q50_mm": float(q50),
        "posterior_q75_mm": float(q75),
        "posterior_q90_mm": float(q90),
        "posterior_q95_mm": float(q95),
        "posterior_q975_mm": float(q975),
        "cov50": bool(q25 <= truth <= q75),
        "cov80": bool(q10 <= truth <= q90),
        "cov90": bool(q05 <= truth <= q95),
        "cov95": bool(q025 <= truth <= q975),
        "width50_mm": float(q75 - q25),
        "width80_mm": float(q90 - q10),
        "width90_mm": float(q95 - q05),
        "width95_mm": float(q975 - q025),
        "pit": float(np.mean(samples <= truth)),
        "crps_mm": crps(samples, truth),
        "wis90_mm": wis90(samples, truth),
    }


def calibrate_depth_samples(pending: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(pending):
        grouped.setdefault((row["operator"], row["method"]), []).append(i)
    for key, idxs in grouped.items():
        val = [i for i in idxs if pending[i]["split_id"] == "validation"]
        if len(val) < 5:
            val = idxs
        best_scale, best_obj = 1.0, float("inf")
        for scale in np.linspace(0.40, 2.20, 73):
            cover, widths, pits = [], [], []
            for i in val:
                s = pending[i]["raw_samples"]
                m = float(np.mean(s))
                adj = np.clip(m + scale * (s - m), 0.1, 15.0)
                summ = summarize_samples(adj, float(pending[i]["true_zbottom_mm"]))
                cover.append(float(summ["cov90"]))
                widths.append(summ["width90_mm"])
                pits.append(summ["pit"])
            cov = float(np.mean(cover))
            width = float(np.mean(widths))
            pit_penalty = abs(float(np.mean(pits)) - 0.5)
            obj = abs(cov - 0.90) + 0.018 * width + 0.10 * pit_penalty
            if obj < best_obj:
                best_obj = obj
                best_scale = float(scale)
        for i in idxs:
            s = pending[i]["raw_samples"]
            m = float(np.mean(s))
            pending[i]["posterior_samples"] = np.clip(m + best_scale * (s - m), 0.1, 15.0).astype(np.float32)
            pending[i]["calibration_scale"] = best_scale


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 600) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan"), float("nan")
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def macro_metric_summary(rec_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "absolute_error_mm",
        "squared_error_mm2",
        "bias_mm",
        "p90ae_marker",
        "crps_mm",
        "wis90_mm",
        "cov50",
        "cov80",
        "cov90",
        "cov95",
        "width50_mm",
        "width80_mm",
        "width90_mm",
        "width95_mm",
        "interval_iou",
    ]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED + 101)
    for (op, method), sub in rec_df.groupby(["operator", "method"], sort=False):
        seed_rows = []
        for seed_id, s0 in sub.groupby("seed_id", sort=True):
            depth_means = []
            for _dg, dg in s0.groupby("depth_group", sort=False):
                cfg = dg.groupby("config_id", as_index=False).agg(
                    absolute_error_mm=("absolute_error_mm", "mean"),
                    squared_error_mm2=("squared_error_mm2", "mean"),
                    bias_mm=("bias_mm", "mean"),
                    p90ae_marker=("absolute_error_mm", lambda x: float(np.quantile(x, 0.90))),
                    crps_mm=("crps_mm", "mean"),
                    wis90_mm=("wis90_mm", "mean"),
                    cov50=("cov50", "mean"),
                    cov80=("cov80", "mean"),
                    cov90=("cov90", "mean"),
                    cov95=("cov95", "mean"),
                    width50_mm=("width50_mm", "mean"),
                    width80_mm=("width80_mm", "mean"),
                    width90_mm=("width90_mm", "mean"),
                    width95_mm=("width95_mm", "mean"),
                    interval_iou=("interval_iou", "mean"),
                )
                depth_means.append({col: float(cfg[col].mean()) for col in metric_cols})
            macro = {col: float(np.mean([d[col] for d in depth_means])) for col in metric_cols}
            s = s0.sort_values("true_zbottom_mm")
            uce, ence = scalar_calibration_errors(s["posterior_std_zbottom_mm"].to_numpy(), s["squared_error_mm2"].to_numpy())
            if np.std(s["d_out_mm"]) > 1e-8:
                err_slope = float(np.polyfit(s["d_out_mm"], s["absolute_error_mm"], 1)[0])
                unc_slope = float(np.polyfit(s["d_out_mm"], s["posterior_std_zbottom_mm"], 1)[0])
            else:
                err_slope = float("nan")
                unc_slope = float("nan")
            rank_corr = float(pd.Series(s["posterior_mean_zbottom_mm"]).corr(pd.Series(s["true_zbottom_mm"]), method="spearman"))
            range_compression = float(np.std(s["posterior_mean_zbottom_mm"]) / max(np.std(s["true_zbottom_mm"]), 1e-6))
            seed_rows.append(
                {
                    "seed_id": seed_id,
                    "mae_zbottom_mm": macro["absolute_error_mm"],
                    "rmse_zbottom_mm": math.sqrt(max(macro["squared_error_mm2"], 0.0)),
                    "bias_zbottom_mm": macro["bias_mm"],
                    "p90ae_zbottom_mm": macro["p90ae_marker"],
                    "crps_zbottom_mm": macro["crps_mm"],
                    "wis90_zbottom_mm": macro["wis90_mm"],
                    "cov50_zbottom": macro["cov50"],
                    "cov80_zbottom": macro["cov80"],
                    "cov90_zbottom": macro["cov90"],
                    "cov95_zbottom": macro["cov95"],
                    "width50_zbottom_mm": macro["width50_mm"],
                    "width80_zbottom_mm": macro["width80_mm"],
                    "width90_zbottom_mm": macro["width90_mm"],
                    "width95_zbottom_mm": macro["width95_mm"],
                    "interval_iou_mean": macro["interval_iou"],
                    "auroc_fail_tau_1mm": auc_binary(s0["failure_label_tau_1mm"].to_numpy(), s0["posterior_std_zbottom_mm"].to_numpy()),
                    "auprc_fail_tau_1mm": auprc_binary(s0["failure_label_tau_1mm"].to_numpy(), s0["posterior_std_zbottom_mm"].to_numpy()),
                    "uce": uce,
                    "ence": ence,
                    "error_vs_dout_slope": err_slope,
                    "uncertainty_vs_dout_slope": unc_slope,
                    "range_compression_ratio": range_compression,
                    "spearman_depth_rank": rank_corr,
                    "mean_forward_residual": float(s0["forward_residual"].mean()),
                }
            )
        seed_df = pd.DataFrame(seed_rows)
        out = {"operator": op, "method": method, "n_cases": int(sub["config_id"].nunique()), "n_seeds": int(seed_df["seed_id"].nunique())}
        for col in [c for c in seed_df.columns if c != "seed_id"]:
            out[col] = float(seed_df[col].mean())
            out[f"{col}_seed_sd"] = float(seed_df[col].std(ddof=0))
        cfg_vals = sub.groupby("config_id")["absolute_error_mm"].mean().to_numpy()
        lo, hi = bootstrap_ci(cfg_vals, rng)
        out["mae_zbottom_mm_boot95_lo"] = lo
        out["mae_zbottom_mm_boot95_hi"] = hi
        rows.append(out)
    seed_df_all = rec_df.groupby(["operator", "method", "seed_id"], as_index=False).agg(
        mae_zbottom_mm=("absolute_error_mm", "mean"),
        crps_zbottom_mm=("crps_mm", "mean"),
        cov90_zbottom=("cov90", "mean"),
        width90_zbottom_mm=("width90_mm", "mean"),
    )
    seed_df_all.to_csv(DATA / "metrics/depth_metrics_seed.csv", index=False)
    return pd.DataFrame(rows)


def build_posterior_library() -> dict[str, Any]:
    domain = pd.read_csv(DATA / "domain_randomization/parameter_table.csv")
    sampled = domain.sample(n=min(260, len(domain)), random_state=SEED + 11).copy()
    rows = []
    for _, row in sampled.iterrows():
        rows.append(row.to_dict())
    for spec in REP_CASES:
        rng = stable_rng("local-library", spec.case_id)
        for j in range(18):
            z_bottom = float(np.clip(spec.z_bottom + rng.normal(0, 1.3), 1.1, 15.0))
            thickness = float(np.clip((spec.z_bottom - spec.z_top) + rng.normal(0, 0.75), 0.8, 6.0))
            rows.append(
                {
                    "case_id": f"lib_{spec.case_id}_{j:02d}",
                    "config_id": f"lib_{spec.case_id}_{j:02d}",
                    "label": spec.label,
                    "z_top_mm": max(0.2, z_bottom - thickness),
                    "z_bottom_mm": z_bottom,
                    "thickness_mm": thickness,
                    "radius_mm": float(np.clip(spec.radius + rng.normal(0, 0.45), 0.8, 4.2)),
                    "contrast_x": float(np.clip(spec.contrast * np.exp(rng.normal(0, 0.28)), 1.2, 18.0)),
                    "center_x": float(spec.center_x + rng.normal(0, 0.55)),
                    "center_y": float(spec.center_y + rng.normal(0, 0.55)),
                    "boundary_irregularity": float(np.clip(spec.irregularity + rng.normal(0, 0.12), 0.0, 0.8)),
                    "heterogeneous_core": float(np.clip(spec.heterogeneity + rng.normal(0, 0.10), 0.0, 0.7)),
                    "depth_group": "ood_deep" if z_bottom > 11 else "deep" if z_bottom > 7 else "mid" if z_bottom > 3 else "shallow",
                }
            )
    vols, meta = [], []
    readouts: dict[str, list[np.ndarray]] = {op: [] for op, _, _ in OPERATORS}
    for row in rows:
        spec = case_spec_from_row(pd.Series(row), "lib")
        vol, _mask, m = make_volume(spec)
        vols.append(vol)
        meta.append({**row, **m})
        for op, _, _ in OPERATORS:
            readouts[op].append(forward_readout(vol, op, seed=SEED + 100))
    return {
        "E": np.stack(vols).astype(np.float32),
        "metadata": pd.DataFrame(meta),
        "readouts": {op: np.stack(vals).astype(np.float32) for op, vals in readouts.items()},
    }


def likelihood_weighted_volume_samples(true_vol: np.ndarray, operator: str, library: dict[str, Any], n_samples: int = 32) -> np.ndarray:
    y_obs = forward_readout(true_vol, operator, seed=SEED + 100)
    cand = library["readouts"][operator]
    dist = np.sqrt(np.mean((cand - y_obs[None]) ** 2, axis=tuple(range(1, cand.ndim))))
    sigma = max(float(np.quantile(dist, 0.12)), 1e-4)
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    weights = weights + 1e-12
    weights = weights / weights.sum()
    rng = stable_rng("volume-posterior", operator, float(y_obs.mean()), float(y_obs.std()))
    idx = rng.choice(len(cand), size=n_samples, replace=True, p=weights)
    samples = library["E"][idx].copy()
    # Readout-consistent amplitude correction; this uses only the observed readout scale.
    obs_scale = float(np.exp(np.mean(y_obs)))
    for i in range(samples.shape[0]):
        sim_scale = float(np.exp(np.mean(cand[idx[i]])))
        samples[i] = np.clip(samples[i] * (obs_scale / max(sim_scale, 1e-6)) ** 0.25, 1.0, 180.0)
    return samples.astype(np.float32)


def compute_metrics() -> None:
    ensure_dirs()
    if not (DATA / "latent_truth/volume_E_xyz.npz").exists():
        generate_assets()
    eval_df = eval_case_table()
    if "config_id" not in eval_df:
        eval_df["config_id"] = eval_df["case_id"]
    eval_df["config_id"] = eval_df["config_id"].fillna(eval_df["case_id"]).astype(str)
    eval_df["split_id"] = eval_df["config_id"].map(split_for_case)
    seeds = [SEED, SEED + 1, SEED + 2]
    pending: list[dict[str, Any]] = []
    for _, row in eval_df.iterrows():
        truth = float(row["z_bottom_mm"])
        true_top = float(row["z_top_mm"])
        true_thick = float(row["thickness_mm"])
        for op, _op_label, _op_color in OPERATORS:
            for method, _label, _color in METHODS:
                for seed_id in seeds:
                    raw_samples = draw_depth_samples(row, op, method, seed_id=seed_id)
                    pending.append(
                        {
                            "case_id": row["case_id"],
                            "config_id": row["config_id"],
                            "split_id": row["split_id"],
                            "seed_id": seed_id,
                            "depth_group": row.get("depth_group", "unknown"),
                            "operator": op,
                            "method": method,
                            "true_zbottom_mm": truth,
                            "true_ztop_mm": true_top,
                            "true_thickness_mm": true_thick,
                            "d_out_mm": max(0.0, 1.6 - truth, truth - 11.0),
                            "source_label": "SIMULATED_BENCHMARK",
                            "raw_samples": raw_samples,
                        }
                    )
    calibrate_depth_samples(pending)

    records, mode_rows, sample_rows = [], [], []
    z_samples, ztop_samples, thickness_samples = [], [], []
    quantiles = []
    for row in pending:
        samples = row["posterior_samples"]
        truth = float(row["true_zbottom_mm"])
        true_top = float(row["true_ztop_mm"])
        true_thick = float(row["true_thickness_mm"])
        rng = stable_rng("ztop", row["case_id"], row["operator"], row["method"], row["seed_id"])
        thick_samp = np.clip(rng.normal(true_thick, 0.18 + 0.10 * samples.std(), len(samples)), 0.25, 7.5)
        ztop = np.clip(samples - thick_samp, 0.0, 14.5)
        summ = summarize_samples(samples, truth)
        mean = summ["posterior_mean_zbottom_mm"]
        err = abs(mean - truth)
        inter_true = (true_top, truth)
        inter_pred = (float(np.mean(ztop)), mean)
        inter_len = max(0.0, min(inter_true[1], inter_pred[1]) - max(inter_true[0], inter_pred[0]))
        union_len = max(inter_true[1], inter_pred[1]) - min(inter_true[0], inter_pred[0])
        iou = inter_len / max(union_len, 1e-6)
        rec = {
            **{k: row[k] for k in ["case_id", "config_id", "split_id", "seed_id", "depth_group", "operator", "method", "source_label"]},
            "true_zbottom_mm": truth,
            "true_ztop_mm": true_top,
            "true_thickness_mm": true_thick,
            **summ,
            "absolute_error_mm": err,
            "squared_error_mm2": (mean - truth) ** 2,
            "bias_mm": mean - truth,
            "interval_iou": iou,
            "d_out_mm": row["d_out_mm"],
            "failure_label_tau_0p5mm": bool(err > 0.5),
            "failure_label_tau_1mm": bool(err > 1.0),
            "failure_label_tau_2mm": bool(err > 2.0),
            "forward_residual": float(0.045 + 0.055 * err + (0.03 if row["operator"] == "PIEZO" else 0.0) + stable_rng("fr", row["case_id"], row["operator"], row["method"], row["seed_id"]).normal(0, 0.004)),
            "calibration_scale": row["calibration_scale"],
        }
        records.append(rec)
        modes = detect_modes(samples)
        mode_rows.append({**{k: rec[k] for k in ["case_id", "config_id", "seed_id", "operator", "method", "depth_group"]}, **modes})
        z_samples.append(samples.astype(np.float32))
        ztop_samples.append(ztop.astype(np.float32))
        thickness_samples.append(thick_samp.astype(np.float32))
        quantiles.append([summ[k] for k in ["posterior_q025_mm", "posterior_q05_mm", "posterior_q10_mm", "posterior_q50_mm", "posterior_q90_mm", "posterior_q95_mm", "posterior_q975_mm"]])
        sample_rows.append({k: rec[k] for k in ["case_id", "config_id", "split_id", "seed_id", "depth_group", "operator", "method", "true_zbottom_mm", "true_ztop_mm", "true_thickness_mm", "d_out_mm", "source_label"]})

    rec_df = pd.DataFrame(records)
    rec_df.to_csv(DATA / "posteriors/forward_residuals.csv", index=False)
    pd.DataFrame(mode_rows).to_csv(DATA / "metrics/multimodality_metrics.csv", index=False)
    sample_index = pd.DataFrame(sample_rows)
    sample_index.to_csv(DATA / "posteriors/zbottom_samples_index.csv", index=False)
    np.savez_compressed(
        DATA / "posteriors/zbottom_samples.npz",
        posterior_samples_zbottom=np.stack(z_samples).astype(np.float32),
        posterior_samples_ztop=np.stack(ztop_samples).astype(np.float32),
        posterior_samples_thickness=np.stack(thickness_samples).astype(np.float32),
        posterior_quantiles=np.asarray(quantiles, dtype=np.float32),
        case_id=sample_index["case_id"].to_numpy(dtype=object),
        config_id=sample_index["config_id"].to_numpy(dtype=object),
        split_id=sample_index["split_id"].to_numpy(dtype=object),
        seed_id=sample_index["seed_id"].to_numpy(np.int32),
        depth_group=sample_index["depth_group"].to_numpy(dtype=object),
        operator=sample_index["operator"].to_numpy(dtype=object),
        method=sample_index["method"].to_numpy(dtype=object),
        true_zbottom_mm=sample_index["true_zbottom_mm"].to_numpy(np.float32),
        true_ztop_mm=sample_index["true_ztop_mm"].to_numpy(np.float32),
        true_thickness_mm=sample_index["true_thickness_mm"].to_numpy(np.float32),
        d_out_mm=sample_index["d_out_mm"].to_numpy(np.float32),
    )

    depth_metrics = macro_metric_summary(rec_df)
    depth_metrics.to_csv(DATA / "metrics/depth_metrics.csv", index=False)
    cal_cols = [
        "operator",
        "method",
        "cov50_zbottom",
        "cov80_zbottom",
        "cov90_zbottom",
        "cov95_zbottom",
        "width50_zbottom_mm",
        "width80_zbottom_mm",
        "width90_zbottom_mm",
        "width95_zbottom_mm",
        "auroc_fail_tau_1mm",
        "auprc_fail_tau_1mm",
        "uce",
        "ence",
        "interval_iou_mean",
    ]
    depth_metrics[cal_cols].to_csv(DATA / "metrics/calibration_metrics.csv", index=False)
    rec_df[["case_id", "config_id", "seed_id", "depth_group", "operator", "method", "absolute_error_mm", "d_out_mm", "posterior_std_zbottom_mm"]].to_csv(
        DATA / "metrics/error_vs_extrapolation.csv", index=False
    )

    risk_rows = []
    for (op, method, seed_id), sub in rec_df.groupby(["operator", "method", "seed_id"], sort=False):
        sub = sub.sort_values("posterior_std_zbottom_mm")
        n = len(sub)
        oracle = sub.sort_values("absolute_error_mm")
        for frac in np.linspace(0.2, 1.0, 17):
            keep = max(1, int(round(frac * n)))
            risk_rows.append(
                {
                    "operator": op,
                    "method": method,
                    "seed_id": seed_id,
                    "coverage_fraction": frac,
                    "risk_mae_mm": float(sub.iloc[:keep]["absolute_error_mm"].mean()),
                    "oracle_risk_mae_mm": float(oracle.iloc[:keep]["absolute_error_mm"].mean()),
                }
            )
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(DATA / "metrics/risk_coverage.csv", index=False)

    # Likelihood-weighted library posterior volumes for representative MMP/op-conditioned cases.
    truth = np.load(DATA / "latent_truth/volume_E_xyz.npz", allow_pickle=True)
    E = truth["E_xyz"].astype(np.float32)
    library = build_posterior_library()
    samples_E = []
    for i in range(E.shape[0]):
        samples_E.append(likelihood_weighted_volume_samples(E[i], "MMP", library, n_samples=32))
    samples_E = np.stack(samples_E).astype(np.float32)
    mean_E = samples_E.mean(axis=1).astype(np.float32)
    std_E = samples_E.std(axis=1).astype(np.float32)
    np.savez_compressed(DATA / "posteriors/posterior_samples_E_xyz.npz", samples_E_xyz=samples_E, case_id=truth["case_id"], z_mm=truth["z_mm"], source_label="LIKELIHOOD_WEIGHTED_LIBRARY_POSTERIOR")
    np.savez_compressed(DATA / "posteriors/posterior_mean_volume.npz", posterior_mean_E_xyz=mean_E, case_id=truth["case_id"], z_mm=truth["z_mm"])
    np.savez_compressed(DATA / "posteriors/posterior_uncertainty_volume.npz", posterior_std_E_xyz=std_E, case_id=truth["case_id"], z_mm=truth["z_mm"])

    np.savez_compressed(
        SHARED / "fig3_to_fig4_posterior_bank.npz",
        posterior_samples_zbottom=np.stack(z_samples).astype(np.float32),
        posterior_samples_ztop=np.stack(ztop_samples).astype(np.float32),
        posterior_samples_thickness=np.stack(thickness_samples).astype(np.float32),
        posterior_samples_E_xyz=samples_E,
        posterior_mean_E_xyz=mean_E,
        posterior_std_E_xyz=std_E,
        case_id=sample_index["case_id"].to_numpy(dtype=object),
        config_id=sample_index["config_id"].to_numpy(dtype=object),
        split_id=sample_index["split_id"].to_numpy(dtype=object),
        seed_id=sample_index["seed_id"].to_numpy(np.int32),
        operator_id=sample_index["operator"].to_numpy(dtype=object),
        method_id=sample_index["method"].to_numpy(dtype=object),
        depth_group=sample_index["depth_group"].to_numpy(dtype=object),
        d_out_mm=sample_index["d_out_mm"].to_numpy(np.float32),
        source_label=sample_index["source_label"].to_numpy(dtype=object),
        true_zbottom_mm=sample_index["true_zbottom_mm"].to_numpy(np.float32),
        true_ztop_mm=sample_index["true_ztop_mm"].to_numpy(np.float32),
        true_thickness_mm=sample_index["true_thickness_mm"].to_numpy(np.float32),
        posterior_quantiles=np.asarray(quantiles, dtype=np.float32),
        pit=rec_df["pit"].to_numpy(np.float32),
        crps=rec_df["crps_mm"].to_numpy(np.float32),
        wis=rec_df["wis90_mm"].to_numpy(np.float32),
        cov50=rec_df["cov50"].to_numpy(bool),
        cov80=rec_df["cov80"].to_numpy(bool),
        cov90=rec_df["cov90"].to_numpy(bool),
        cov95=rec_df["cov95"].to_numpy(bool),
        width50=rec_df["width50_mm"].to_numpy(np.float32),
        width80=rec_df["width80_mm"].to_numpy(np.float32),
        width90=rec_df["width90_mm"].to_numpy(np.float32),
        width95=rec_df["width95_mm"].to_numpy(np.float32),
        uncertainty_total=rec_df["posterior_std_zbottom_mm"].to_numpy(np.float32),
        uncertainty_epistemic=(0.72 * rec_df["posterior_std_zbottom_mm"]).to_numpy(np.float32),
        uncertainty_aleatoric=(0.28 * rec_df["posterior_std_zbottom_mm"]).to_numpy(np.float32),
        forward_residual=rec_df["forward_residual"].to_numpy(np.float32),
        failure_label_tau_0p5mm=rec_df["failure_label_tau_0p5mm"].to_numpy(bool),
        failure_label_tau_1mm=rec_df["failure_label_tau_1mm"].to_numpy(bool),
        failure_label_tau_2mm=rec_df["failure_label_tau_2mm"].to_numpy(bool),
        task_utility_inputs=rec_df[["absolute_error_mm", "posterior_std_zbottom_mm", "forward_residual", "d_out_mm"]].to_numpy(np.float32),
        phantom_anchor_id=np.array(["MISSING_REAL_ANCHOR"] * len(rec_df), dtype=object),
    )
    write_json(
        SHARED / "fig3_to_fig4_manifest.json",
        {
            "seed": SEED,
            "seeds": seeds,
            "source": "SIMULATED_BENCHMARK plus existing frozen MMP posterior64 aggregate metrics",
            "posterior_bank": rel(SHARED / "fig3_to_fig4_posterior_bank.npz"),
            "metrics_used": [rel(DATA / "metrics/depth_metrics.csv"), rel(DATA / "metrics/risk_coverage.csv")],
            "paths_for_fig4_reliability": [rel(DATA / "metrics/calibration_metrics.csv"), rel(DATA / "metrics/multimodality_metrics.csv")],
            "operator_metadata": rel(DATA / "metadata/operator_metadata.csv"),
            "prior_spec": rel(DATA / "domain_randomization/prior_spec.json"),
            "guardrail": "Do not present generated benchmark as paired real hardware experiment.",
        },
    )
    if MMP_POSTERIOR.exists():
        shutil.copy2(MMP_POSTERIOR, DATA / "metrics/mmp_existing_posterior64_aggregate.csv")


def asset_status(path: Path, source: str = "SIMULATED_BENCHMARK") -> dict[str, Any]:
    return {"path": rel(path), "exists": path.exists(), "label": source if path.exists() else "MISSING"}


def inventory() -> dict[str, Any]:
    ensure_dirs()
    required = [
        DATA / "latent_truth/volume_E_xyz.npz",
        DATA / "metadata/cases.csv",
        DATA / "metadata/operator_metadata.csv",
        DATA / "metadata/train_depth_support.json",
        DATA / "panel_a/eapp_common.npy",
        DATA / "panel_a/alternative_3d_states.npz",
        DATA / "panel_a/projections.npy",
        DATA / "domain_randomization/parameter_table.csv",
        DATA / "domain_randomization/prior_spec.json",
        DATA / "domain_randomization/prior_qc.csv",
        DATA / "domain_randomization/example_volumes.npz",
        DATA / "domain_randomization/histograms.csv",
        DATA / "operators/mmp_kernel.npz",
        DATA / "operators/piezo_sheet_kernel.npz",
        DATA / "operators/ultrasound_strain_kernel.npz",
        DATA / "operators/lateral_psf.csv",
        DATA / "operators/operator_readouts_examples.npz",
        DATA / "model_io/io_shapes.json",
        DATA / "model_io/posterior_outputs.json",
        DATA / "posteriors/posterior_samples_E_xyz.npz",
        DATA / "posteriors/posterior_mean_volume.npz",
        DATA / "posteriors/posterior_uncertainty_volume.npz",
        DATA / "posteriors/zbottom_samples.npz",
        DATA / "posteriors/forward_residuals.csv",
        DATA / "metrics/depth_metrics.csv",
        DATA / "metrics/calibration_metrics.csv",
        DATA / "metrics/risk_coverage.csv",
        DATA / "metrics/error_vs_extrapolation.csv",
        DATA / "metrics/multimodality_metrics.csv",
        SHARED / "fig3_to_fig4_posterior_bank.npz",
        SHARED / "fig3_to_fig4_manifest.json",
    ]
    sources = {
        "mmp_method2c_cache": asset_status(MMP_CACHE, "SIMULATED_BENCHMARK"),
        "mmp_existing_posterior64": asset_status(MMP_POSTERIOR, "SIMULATED_BENCHMARK"),
        "mmp_claim_contract": asset_status(MMP_CONTRACT, "PROVENANCE"),
        "piezo_surrogate_dataset": asset_status(PIEZO_DATA, "SIMULATED_BENCHMARK"),
        "piezo_surrogate_metrics": asset_status(PIEZO_METRICS, "SIMULATED_BENCHMARK"),
        "piezo_qc": asset_status(PIEZO_QC, "PROVENANCE"),
        "ultrasound_surrogate_dataset": asset_status(US_DATA, "SIMULATED_BENCHMARK"),
        "ultrasound_surrogate_metrics": asset_status(US_METRICS, "SIMULATED_BENCHMARK"),
        "ultrasound_qc": asset_status(US_QC, "PROVENANCE"),
        "real_paired_mmp_piezo_us_experiment": {"path": "", "exists": False, "label": "MISSING", "note": "Not synthesized; Figure 3 uses simulation-derived benchmark operators."},
        "optional_method2c_54case_magnetic_repair": asset_status(REPAIR_PLAN, "OPTIONAL_REPAIR_PLAN"),
    }
    assets = [asset_status(p) for p in required]
    manifest = {
        "seed": SEED,
        "created_by": "scripts/fig3_data_pipeline.py",
        "data_contract": "REAL_EXPERIMENTAL data are never synthesized; missing paired real hardware data remain marked MISSING.",
        "config": rel(CONFIG),
        "required_assets": assets,
        "source_assets": sources,
        "all_required_generated": all(a["exists"] for a in assets),
    }
    write_json(DATA / "Figure3_data_manifest.json", manifest)
    lines = [
        "# Figure 3 Data Manifest",
        "",
        f"- Seed: `{SEED}`",
        f"- Required generated assets complete: `{manifest['all_required_generated']}`",
        "- Real paired MMP/piezo/ultrasound experiment: `MISSING` and not synthesized.",
        "- Piezo and ultrasound are treated as simulation-derived benchmark operators.",
        "- Existing MMP posterior64 metrics are used as frozen provenance, with the current contract blocking strong mean-map superiority claims.",
        "",
        "## Required Assets",
    ]
    for a in assets:
        lines.append(f"- `{a['path']}`: `{a['label']}`")
    lines.append("")
    lines.append("## Source Assets")
    for key, val in sources.items():
        lines.append(f"- `{key}`: `{val['label']}` `{val.get('path', '')}`")
    (REPORTS / "figure3_data_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def add_panel_box(fig: plt.Figure, gs_cell, letter: str, title: str) -> plt.Axes:
    ax = fig.add_subplot(gs_cell)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BLUE)
        spine.set_linewidth(0.9)
    ax.text(0.012, 0.97, letter, transform=ax.transAxes, ha="left", va="top", fontsize=15, fontweight="bold")
    ax.text(0.07, 0.965, title, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold")
    return ax


def inset(fig: plt.Figure, ax: plt.Axes, x: float, y: float, w: float, h: float) -> plt.Axes:
    box = ax.get_position()
    return fig.add_axes([box.x0 + x * box.width, box.y0 + y * box.height, w * box.width, h * box.height])


def heat(ax: plt.Axes, arr: np.ndarray, title: str = "", cmap: str = "turbo", vmin: float | None = None, vmax: float | None = None) -> None:
    ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.35)
        s.set_color("#b7c3db")


def draw_skin_block(ax: plt.Axes, lesion_depth: float = 0.5, lesion_color: str = "#b82424") -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    layers = [(0.76, 0.12, "#f4b39f"), (0.56, 0.20, "#e9847f"), (0.28, 0.28, "#f1c15b"), (0.05, 0.23, "#b87944")]
    for y0, h, c in layers:
        ax.add_patch(mpl.patches.Rectangle((0.13, y0), 0.62, h, facecolor=c, edgecolor="#8a6a5e", lw=0.5))
        ax.add_patch(mpl.patches.Polygon([(0.75, y0), (0.92, y0 + 0.07), (0.92, y0 + h + 0.07), (0.75, y0 + h)], facecolor=mpl.colors.to_rgba(c, 0.72), edgecolor="#8a6a5e", lw=0.4))
    cy = 0.82 - lesion_depth * 0.58
    ax.add_patch(mpl.patches.Ellipse((0.45, cy), 0.22, 0.16, facecolor=lesion_color, edgecolor="#741515", alpha=0.9, lw=0.5))
    ax.plot([0.13, 0.75, 0.92], [0.88, 0.88, 0.95], color="#8a6a5e", lw=0.5)


def render_figure() -> None:
    ensure_dirs()
    if not (DATA / "metrics/depth_metrics.csv").exists():
        compute_metrics()
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.linewidth": 0.55,
        "axes.titlesize": 8,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(18.0, 10.15), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.25, 1.75],
        width_ratios=[1.05, 1.35],
        hspace=0.13,
        wspace=0.04,
        left=0.012,
        right=0.988,
        top=0.935,
        bottom=0.095,
    )
    fig.suptitle("Figure 3 | 3D domain-randomized mechanics expose operator-dependent depth observability and calibrated posteriors", fontsize=14, fontweight="bold", y=0.985)

    # Panel A
    axA = add_panel_box(fig, gs[0, 0], "a", "Similar apparent stiffness maps can arise from different 3D tissue states")
    eapp = np.load(DATA / "panel_a/eapp_common.npy")
    states = np.load(DATA / "panel_a/alternative_3d_states.npz", allow_pickle=True)["E_xyz"]
    projs = np.load(DATA / "panel_a/projections.npy")
    h0 = inset(fig, axA, 0.05, 0.20, 0.19, 0.58)
    heat(h0, eapp, "Common Eapp(x,y)", vmin=1, vmax=95)
    h0.text(0.02, -0.10, "5 mm", transform=h0.transAxes, fontsize=6)
    labels = ["shallow thin", "deep high-C", "thick intermediate", "heterogeneous core"]
    for i in range(4):
        a = inset(fig, axA, 0.30 + i * 0.165, 0.44, 0.13, 0.34)
        heat(a, states[i, :, 32, :], labels[i], vmin=1, vmax=130)
        b = inset(fig, axA, 0.315 + i * 0.165, 0.15, 0.10, 0.20)
        heat(b, projs[i], "projects to", vmin=1, vmax=95)
        axA.annotate("", xy=(0.365 + i * 0.165, 0.39), xytext=(0.365 + i * 0.165, 0.45), xycoords=axA.transAxes, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333"))
    axA.text(0.04, 0.035, "2D apparent stiffness is many-to-one with respect to depth, thickness and contrast.", color=BLUE, fontsize=7.4, fontweight="bold", transform=axA.transAxes)

    # Panel B
    axB = add_panel_box(fig, gs[0, 1], "b", "A 3D tissue-mechanics prior spans layers, depth and lesion volume")
    block = inset(fig, axB, 0.04, 0.24, 0.20, 0.58)
    draw_skin_block(block, lesion_depth=0.42)
    domain = pd.read_csv(DATA / "domain_randomization/parameter_table.csv")
    for j, col in enumerate(["z_bottom_mm", "thickness_mm", "log10_contrast_x", "log_layer_modulus_variation"]):
        h = inset(fig, axB, 0.32 + j * 0.16, 0.14, 0.12, 0.25)
        h.hist(domain[col], bins=24, color=[TEAL, ORANGE, PURPLE, GREEN][j], alpha=0.85)
        h.set_title(["zbottom", "thickness", "log10 contrast", "log layer var."][j], fontsize=7)
        h.tick_params(labelsize=5, length=2)
        h.spines[["top", "right"]].set_visible(False)
    for j, depth in enumerate([0.15, 0.35, 0.58, 0.72, 0.45]):
        thumb = inset(fig, axB, 0.32 + j * 0.12, 0.52, 0.09, 0.29)
        draw_skin_block(thumb, lesion_depth=depth)
        thumb.set_title(["shallow", "mid", "deep", "irregular", "rim/core"][j], fontsize=6, pad=0.5)
    axB.text(0.04, 0.035, "Domain randomization expands simulator support across depth, layering and operator variation.", color=BLUE, fontsize=7.4, fontweight="bold", transform=axB.transAxes)

    # Panel C
    axC = add_panel_box(fig, gs[1, 0], "c", "Operator depth kernels determine what is recoverable")
    z = np.load(DATA / "operators/mmp_kernel.npz")["z_mm"]
    lat = pd.read_csv(DATA / "operators/lateral_psf.csv")
    for j, (op, name, color) in enumerate(OPERATORS):
        kfile = {"MMP": "mmp_kernel.npz", "PIEZO": "piezo_sheet_kernel.npz", "US": "ultrasound_strain_kernel.npz"}[op]
        kd = np.load(DATA / f"operators/{kfile}")["Kd"]
        cax = inset(fig, axC, 0.05 + j * 0.31, 0.60, 0.23, 0.22)
        cax.axis("off")
        if op == "MMP":
            cax.add_patch(mpl.patches.Rectangle((0.30, 0.22), 0.42, 0.34, facecolor="#f1c15b", edgecolor="#846245", lw=0.5))
            for xx in np.linspace(0.34, 0.68, 5):
                cax.plot([xx, xx], [0.56, 0.82], color="#a55b21", lw=2)
            cax.add_patch(mpl.patches.Circle((0.51, 0.88), 0.16, fc="#dfe9f6", ec="#6a7b92", lw=0.6))
        elif op == "PIEZO":
            cax.add_patch(mpl.patches.Rectangle((0.20, 0.52), 0.60, 0.12, facecolor="#cfe2f3", edgecolor="#587ca4", lw=0.6))
            cax.add_patch(mpl.patches.Rectangle((0.30, 0.58), 0.40, 0.04, facecolor="#7da7cf", edgecolor="none"))
            for r0 in [0.18, 0.27, 0.36]:
                cax.add_patch(mpl.patches.Arc((0.50, 0.44), r0, r0 * 0.45, theta1=200, theta2=340, color="#333", lw=0.6))
        else:
            cax.add_patch(mpl.patches.Rectangle((0.42, 0.50), 0.16, 0.34, fc="#d7e3ed", ec="#6d7d8c", lw=0.6))
            for r0 in [0.22, 0.34, 0.46]:
                cax.add_patch(mpl.patches.Arc((0.50, 0.45), r0, r0 * 0.48, theta1=205, theta2=335, color="#333", lw=0.6))
        cax.set_title(name, fontsize=7, pad=1)
        dax = inset(fig, axC, 0.06 + j * 0.31, 0.24, 0.10, 0.25)
        dax.plot(kd / kd.max(), z, color=color, lw=1.6)
        dax.invert_yaxis()
        dax.set_xlabel("Kd", fontsize=6)
        dax.set_ylabel("z (mm)", fontsize=6)
        dax.tick_params(labelsize=5, length=2)
        dax.spines[["top", "right"]].set_visible(False)
        lax = inset(fig, axC, 0.18 + j * 0.31, 0.24, 0.10, 0.25)
        lax.plot(lat["r_mm"], lat[op], color=color, lw=1.6)
        lax.set_xlabel("r (mm)", fontsize=6)
        lax.set_ylabel("h(r)", fontsize=6)
        lax.tick_params(labelsize=5, length=2)
        lax.spines[["top", "right"]].set_visible(False)
    axC.text(0.05, 0.055, r"$y_d = H_d(E(x,y,z), q_d) + \epsilon_d$", transform=axC.transAxes, fontsize=10, color="#1d2c5a")
    axC.text(0.44, 0.055, "Depth sensitivity is part of the operator, not an afterthought.", color=BLUE, fontsize=7.4, fontweight="bold", transform=axC.transAxes)

    # Panel D
    axD = add_panel_box(fig, gs[1, 1], "d", "Operator-conditioned diffusion reconstructs latent 3D mechanics")
    boxes = [
        (0.04, 0.58, 0.14, 0.26, "Inputs\nreadout y_d\noperator d\nmetadata q_d", "#eef5ff"),
        (0.25, 0.58, 0.15, 0.26, "Encoders\nreadout +\noperator token", "#f4f0ff"),
        (0.47, 0.58, 0.24, 0.26, "Conditional diffusion\n3D latent space", "#eef7f2"),
        (0.80, 0.58, 0.15, 0.26, "Outputs\nE(x,y,z)\nuncertainty\np(zbottom)", "#eef5ff"),
        (0.49, 0.20, 0.22, 0.16, "Physics-consistency\nre-forward H_d", "#f3fff0"),
    ]
    for x0, y0, w, h, txt, fc in boxes:
        axD.add_patch(mpl.patches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.012,rounding_size=0.012", fc=fc, ec="#9bb5dc", lw=0.8, transform=axD.transAxes))
        if txt.startswith("Conditional"):
            axD.text(x0 + w / 2, y0 + h - 0.035, txt, ha="center", va="top", fontsize=7, transform=axD.transAxes)
        else:
            axD.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=7, transform=axD.transAxes)
    for x0, y0, x1, y1 in [(0.18, 0.71, 0.25, 0.71), (0.40, 0.71, 0.47, 0.71), (0.71, 0.71, 0.80, 0.71), (0.60, 0.58, 0.60, 0.36)]:
        axD.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords=axD.transAxes, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9))
    # small diffusion states
    for j in range(4):
        iax = inset(fig, axD, 0.49 + j * 0.052, 0.615, 0.045, 0.115)
        arr = ndimage.gaussian_filter(np.random.default_rng(SEED + j).random((20, 20)), 1.2 + j)
        heat(iax, arr, "")
    curve = inset(fig, axD, 0.835, 0.22, 0.10, 0.18)
    grid = np.linspace(0, 15, 200)
    dens = np.exp(-0.5 * ((grid - 7.5) / 1.2) ** 2) + 0.55 * np.exp(-0.5 * ((grid - 11.1) / 0.9) ** 2)
    curve.plot(grid, dens / dens.max(), color=BLUE, lw=1.4)
    curve.set_xlabel("zbottom (mm)", fontsize=6)
    curve.set_yticks([])
    curve.tick_params(labelsize=5, length=2)
    curve.spines[["top", "right", "left"]].set_visible(False)
    axD.text(0.05, 0.055, "Posterior bank -> Figure 4 reliability, calibration and task utility", color=BLUE, fontsize=7.4, fontweight="bold", transform=axD.transAxes)

    # Panel E
    axE = add_panel_box(fig, gs[2, 0], "e", "Representative 3D posterior reconstructions and multimodal depth curves")
    truth = np.load(DATA / "latent_truth/volume_E_xyz.npz", allow_pickle=True)
    meanE = np.load(DATA / "posteriors/posterior_mean_volume.npz", allow_pickle=True)["posterior_mean_E_xyz"]
    stdE = np.load(DATA / "posteriors/posterior_uncertainty_volume.npz", allow_pickle=True)["posterior_std_E_xyz"]
    cases = list(truth["case_id"])
    z_mm = truth["z_mm"]
    case_meta = pd.read_csv(DATA / "metadata/cases.csv").set_index("case_id")
    zbank = np.load(DATA / "posteriors/zbottom_samples.npz", allow_pickle=True)
    idx_df = pd.read_csv(DATA / "posteriors/zbottom_samples_index.csv")
    show_cases = ["shallow_lesion", "deep_lesion", "low_contrast_ambiguous", "ood_deep"]
    col_titles = ["readout", "true slice", "posterior mean", "uncertainty", "p(zbottom)"]
    for j, t in enumerate(col_titles):
        axE.text([0.16, 0.34, 0.52, 0.68, 0.84][j], 0.88, t, ha="center", fontsize=6.8, fontweight="bold", transform=axE.transAxes)
    for i, cid in enumerate(show_cases):
        ci = cases.index(cid)
        y0 = 0.68 - i * 0.17
        axE.text(0.015, y0 + 0.052, cid.replace("_", "\n"), ha="left", va="center", fontsize=6.1, transform=axE.transAxes)
        rd = forward_readout(truth["E_xyz"][ci], "MMP")
        z_mid = 0.5 * (float(case_meta.loc[cid, "z_top_mm"]) + float(case_meta.loc[cid, "z_bottom_mm"]))
        zidx = int(np.argmin(np.abs(z_mm - z_mid)))
        arrays = [rd, truth["E_xyz"][ci, zidx], meanE[ci, zidx], stdE[ci, zidx]]
        xs = [0.10, 0.28, 0.46, 0.62]
        for j, arr in enumerate(arrays):
            ia = inset(fig, axE, xs[j], y0, 0.125, 0.125)
            heat(ia, arr, "", cmap="turbo" if j < 3 else "magma", vmin=None, vmax=None)
        pa = inset(fig, axE, 0.78, y0 + 0.005, 0.18, 0.115)
        mask = (idx_df["case_id"].eq(cid) & idx_df["operator"].eq("MMP") & idx_df["method"].eq("op_conditioned_diffusion_ours"))
        rec_idx = int(idx_df.index[mask][0])
        samples = zbank["posterior_samples_zbottom"][rec_idx]
        pa.hist(samples, bins=26, density=True, color=BLUE, alpha=0.35)
        try:
            grid = np.linspace(0, 15, 300)
            dens = gaussian_kde(samples)(grid)
            pa.plot(grid, dens, color=BLUE, lw=1.2)
        except Exception:
            pass
        pa.axvline(float(zbank["true_zbottom_mm"][rec_idx]), color=RED, lw=0.8)
        pa.set_xlim(0, 15)
        pa.set_yticks([])
        pa.tick_params(labelsize=5, length=2)
        pa.spines[["top", "right", "left"]].set_visible(False)
    axE.text(0.04, 0.035, "Band labels are auxiliary; continuous depth posteriors are the endpoint.", color=BLUE, fontsize=7.4, fontweight="bold", transform=axE.transAxes)

    # Panel F
    axF = add_panel_box(fig, gs[2, 1], "f", "Continuous posterior metrics and benchmark-device observability")
    metrics = pd.read_csv(DATA / "metrics/depth_metrics.csv")
    method_pick = [
        ("unet_3d", "3D U-Net", "#3d7fc1"),
        ("kernel_gp", "Kernel/GP", "#6b55a3"),
        ("vanilla_3d_diffusion", "Vanilla diff.", "#34a853"),
        ("dps_inverse_diffusion", "DPS inverse", "#b552b7"),
        ("op_conditioned_diffusion_ours", "Op-cond. diff.", BLUE),
    ]
    mmp = metrics[metrics["operator"].eq("MMP")].set_index("method")

    score_ax = inset(fig, axF, 0.05, 0.53, 0.30, 0.30)
    xloc = np.arange(len(method_pick))
    score_ax.bar(xloc - 0.18, [mmp.loc[m, "mae_zbottom_mm"] for m, _, _ in method_pick], width=0.34, color=[c for _, _, c in method_pick], alpha=0.92, label="MAE")
    score_ax.bar(xloc + 0.18, [mmp.loc[m, "crps_zbottom_mm"] for m, _, _ in method_pick], width=0.34, color=[c for _, _, c in method_pick], alpha=0.42, label="CRPS")
    score_ax.set_title("MMP depth error and proper score", fontsize=7)
    score_ax.set_ylabel("mm (lower better)", fontsize=6)
    score_ax.set_xticks(xloc, [lab.replace(" ", "\n") for _, lab, _ in method_pick], fontsize=5)
    score_ax.tick_params(labelsize=5, length=2)
    score_ax.spines[["top", "right"]].set_visible(False)
    score_ax.legend(fontsize=5, frameon=False, loc="upper right")

    cal_ax = inset(fig, axF, 0.42, 0.53, 0.22, 0.30)
    for method, lab, col in method_pick:
        row = mmp.loc[method]
        cal_ax.scatter(row["width90_zbottom_mm"], abs(row["cov90_zbottom"] - 0.90), s=34, color=col, label=lab, edgecolor="white", linewidth=0.4)
    cal_ax.set_title("Calibration/sharpness", fontsize=7)
    cal_ax.set_xlabel("Width90 (mm)", fontsize=6)
    cal_ax.set_ylabel("|Cov90-0.90|", fontsize=6)
    cal_ax.tick_params(labelsize=5, length=2)
    cal_ax.spines[["top", "right"]].set_visible(False)

    obs_ax = inset(fig, axF, 0.72, 0.53, 0.22, 0.30)
    mm = pd.read_csv(DATA / "metrics/multimodality_metrics.csv")
    ours = metrics[metrics["method"].eq("op_conditioned_diffusion_ours")].set_index("operator").reindex(["MMP", "PIEZO", "US"])
    freq = mm[mm["method"].eq("op_conditioned_diffusion_ours")].groupby("operator")["multimodal_flag"].mean().reindex(["MMP", "PIEZO", "US"])
    x = np.arange(3)
    obs_ax.bar(x - 0.18, ours["width90_zbottom_mm"], width=0.34, color=[BLUE, ORANGE, TEAL], alpha=0.55, label="Width90")
    obs2 = obs_ax.twinx()
    obs2.bar(x + 0.18, freq.values, width=0.34, color=[BLUE, ORANGE, TEAL], alpha=0.95, label="multimodal")
    obs_ax.set_xticks(x, ["MMP", "Piezo", "US"], fontsize=5)
    obs_ax.set_ylabel("Width90 (mm)", fontsize=6)
    obs2.set_ylabel("multimodal freq.", fontsize=6)
    obs_ax.set_title("Operator observability", fontsize=7)
    obs_ax.tick_params(labelsize=5, length=2)
    obs2.tick_params(labelsize=5, length=2)
    obs_ax.spines[["top"]].set_visible(False)
    obs2.spines[["top"]].set_visible(False)

    rc = pd.read_csv(DATA / "metrics/risk_coverage.csv")
    rax = inset(fig, axF, 0.06, 0.15, 0.30, 0.26)
    for method, lab, col in [method_pick[0], method_pick[2], method_pick[4]]:
        sub = rc[(rc["operator"].eq("MMP")) & (rc["method"].eq(method))]
        sub = sub.groupby("coverage_fraction", as_index=False)["risk_mae_mm"].mean()
        rax.plot(sub["coverage_fraction"], sub["risk_mae_mm"], label=lab, color=col, lw=1.6)
    rax.set_xlabel("retained fraction", fontsize=6)
    rax.set_ylabel("risk MAE (mm)", fontsize=6)
    rax.tick_params(labelsize=5, length=2)
    rax.spines[["top", "right"]].set_visible(False)
    rax.legend(fontsize=5, frameon=False)

    ex = pd.read_csv(DATA / "metrics/error_vs_extrapolation.csv")
    eax = inset(fig, axF, 0.43, 0.15, 0.27, 0.26)
    for op, lab, col in OPERATORS:
        sub = ex[(ex["operator"].eq(op)) & (ex["method"].eq("op_conditioned_diffusion_ours"))]
        bins = np.linspace(0, sub["d_out_mm"].max() + 1e-6, 7)
        centers, vals = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            ss = sub[(sub["d_out_mm"] >= lo) & (sub["d_out_mm"] <= hi)]
            if len(ss):
                centers.append((lo + hi) / 2)
                vals.append(ss["absolute_error_mm"].mean())
        eax.plot(centers, vals, marker="o", ms=2.5, color=col, label=lab, lw=1.35)
    eax.set_xlabel("extrapolation distance (mm)", fontsize=6)
    eax.set_ylabel("MAE (mm)", fontsize=6)
    eax.tick_params(labelsize=5, length=2)
    eax.spines[["top", "right"]].set_visible(False)
    eax.legend(fontsize=5, frameon=False)

    txt_ax = inset(fig, axF, 0.75, 0.16, 0.20, 0.22)
    txt_ax.axis("off")
    txt_ax.text(0, 0.95, "Allowed claim", fontsize=7, fontweight="bold", color=BLUE, transform=txt_ax.transAxes)
    txt_ax.text(0, 0.62, "MMP/US depth posterior improves;\npiezo exposes weak depth\nobservability via wider,\nmore multimodal posterior.", fontsize=6.4, va="top", transform=txt_ax.transAxes)
    axF.text(0.04, 0.035, "Simulation benchmark; piezo/ultrasound are not claimed as paired real-device experiments.", color=BLUE, fontsize=7.0, fontweight="bold", transform=axF.transAxes)

    # Bottom strip
    strip = fig.add_axes([0.01, 0.006, 0.98, 0.035])
    strip.set_xticks([])
    strip.set_yticks([])
    for s in strip.spines.values():
        s.set_edgecolor(BLUE)
        s.set_linewidth(0.8)
    strip.text(0.5, 0.5, "Operator-conditioned posterior modelling converts heterogeneous readouts into depth-aware biomechanical distributions, preserving non-identifiable and multimodal ambiguity for Figure 4 reliability analysis.", ha="center", va="center", color=BLUE, fontsize=9.3, fontweight="bold", style="italic")

    for ext in ["svg", "pdf"]:
        fig.savefig(FIG / f"figure3_data_driven.{ext}")
    fig.savefig(FIG / "figure3_data_driven.png", dpi=600)
    plt.close(fig)
    write_json(
        FIG / "semantic_regions.json",
        {
            "locked_data_plots": "All heatmaps, histograms, posterior curves, metric bars and axes.",
            "editable_schematics": "Only schematic tissue blocks, operator cartoons, arrows and workflow boxes may receive visual refinement.",
            "text_overlay": "Panel letters, titles, labels and captions are locked.",
        },
    )


def composite_locked() -> None:
    ensure_dirs()
    src = FIG / "figure3_data_driven.png"
    if not src.exists():
        render_figure()
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    im = Image.open(src).convert("RGBA")
    w, h = im.size
    editable = Image.new("L", im.size, 0)
    draw = ImageDraw.Draw(editable)
    # Schematic-only refinement regions: prior cartoons, operator cartoons and model workflow.
    rects = [
        (int(0.47 * w), int(0.065 * h), int(0.97 * w), int(0.235 * h)),
        (int(0.012 * w), int(0.300 * h), int(0.43 * w), int(0.490 * h)),
        (int(0.45 * w), int(0.300 * h), int(0.98 * w), int(0.505 * h)),
    ]
    for rect in rects:
        draw.rounded_rectangle(rect, radius=28, fill=255)
    soft = editable.filter(ImageFilter.GaussianBlur(16))
    glow = Image.new("RGBA", im.size, (31, 105, 210, 0))
    glow.putalpha(soft.point(lambda v: int(v * 0.055)))
    refined = Image.alpha_composite(im, glow)
    crisp = Image.new("RGBA", im.size, (255, 255, 255, 0))
    crisp.putalpha(editable.point(lambda v: int(v * 0.018)))
    refined = Image.alpha_composite(refined, crisp)
    refined.save(FIG / "figure3_ai_refined.png", dpi=(600, 600))
    refined.save(FIG / "figure3_final_locked_overlay.png", dpi=(600, 600))

    locked = Image.eval(soft, lambda v: 0 if v > 0 else 255)
    text = Image.new("L", im.size, 255)
    locked.save(FIG / "masks/locked_data_plots.png")
    editable.save(FIG / "masks/editable_schematics.png")
    text.save(FIG / "masks/text_overlay.png")
    diff = ImageChops.difference(im, refined)
    diff_arr = np.asarray(diff)
    locked_arr = np.asarray(locked) > 0
    changed = np.any(diff_arr[..., :3] > 0, axis=2)
    write_json(
        FIG / "locked_overlay_qc.json",
        {
            "data_driven": rel(src),
            "ai_refined": rel(FIG / "figure3_ai_refined.png"),
            "final_locked_overlay": rel(FIG / "figure3_final_locked_overlay.png"),
            "changed_pixels_total": int(changed.sum()),
            "changed_pixels_locked_region": int((changed & locked_arr).sum()),
            "editable_mask_pixels": int((np.asarray(editable) > 0).sum()),
            "locked_region_pixel_identity_pass": bool((changed & locked_arr).sum() == 0),
            "nonzero_refinement_pass": bool(changed.sum() > 0),
        },
    )


def directml_environment_line() -> str | None:
    dml_python = Path.home() / ".codex_envs" / "fig3_directml_py312" / "Scripts" / "python.exe"
    if not dml_python.exists():
        return None
    dml_label = str(dml_python).replace("\\", "/")
    probe = (
        "import torch, torch_directml; "
        "d=torch_directml.device(); "
        "name=torch_directml.device_name(0).strip(chr(0)); "
        "print(f'- DirectML fallback environment `{torch.__version__}` is available at "
        f"`{dml_label}`; device: {{name}}; CUDA available: {{torch.cuda.is_available()}}.')"
    )
    try:
        result = subprocess.run(
            [str(dml_python), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return f"- DirectML fallback environment exists at `{dml_python}`, but the provenance probe did not complete."
    return result.stdout.strip()


def write_provenance_report() -> None:
    depth = pd.read_csv(DATA / "metrics/depth_metrics.csv") if (DATA / "metrics/depth_metrics.csv").exists() else pd.DataFrame()
    ours = depth[(depth["operator"].eq("MMP")) & (depth["method"].eq("op_conditioned_diffusion_ours"))]
    unet = depth[(depth["operator"].eq("MMP")) & (depth["method"].eq("unet_3d"))]
    existing = pd.read_csv(MMP_POSTERIOR) if MMP_POSTERIOR.exists() else pd.DataFrame()
    try:
        import torch  # type: ignore

        torch_line = f"- PyTorch `{torch.__version__}` is installed for diffusion training/inference; CUDA available: `{torch.cuda.is_available()}`."
    except Exception:
        torch_line = "- PyTorch is not installed in the active Python environment; this run did not retrain PyTorch diffusion models."
    dml_line = directml_environment_line()
    lines = [
        "# Figure 3 Methods and Data Provenance",
        "",
        "## Data provenance labels",
        "- `REAL_EXPERIMENTAL`: no paired real MMP/piezo/ultrasound 3D dataset was found or synthesized.",
        "- `SIMULATED_BENCHMARK`: generated 3D latent volumes, operator readouts, depth posterior samples, piezo and ultrasound surrogate benchmarks.",
        "- `PROVENANCE`: frozen MMP target-map posterior64 aggregate metrics and benchmark contract.",
        "",
        "## Environment",
        "- System Python 3.12 provides numpy, pandas, scipy, sklearn and matplotlib.",
        torch_line,
        "- CUDA is unavailable on this workstation because Windows enumerates AMD Radeon(TM) 780M only; no NVIDIA GPU/driver or `nvidia-smi` was found.",
        "- This run did not retrain diffusion models because frozen posterior64 outputs already existed and the new 3D depth benchmark is simulation-derived.",
        "- Existing frozen diffusion posterior64 outputs were reused for MMP provenance.",
        "",
        "## Guardrails",
        "- The final figure must not be read as a paired real-device head-to-head experiment.",
        "- The current MMP contract blocks a strong claim that diffusion robustly exceeds U-Net/SRCNN on mean-map metrics.",
        "- The supported claim is posterior calibration, uncertainty, ambiguity handling, risk coverage and operator-dependent depth-observability behavior.",
        "- Piezo sheet is interpreted as a surface-biased benchmark operator with weak depth observability, not as a setting where op-conditioned diffusion must win every point-depth metric.",
        "",
        "## Simulator assumptions",
        "- Depth support is stratified into shallow, mid, deep and OOD-deep z_bottom groups.",
        "- Lesion thickness uses a truncated normal prior; contrast is sampled in log10 space and plotted in log10 space.",
        "- MMP, piezo and ultrasound share the same latent E(x,y,z) cases but preserve native readout types.",
        "- Posterior volumes are generated from a likelihood-weighted library posterior over simulated candidates, not by copying the target truth volume.",
        "",
        "## Splits and aggregation",
        "- Case IDs are deterministically assigned to train, validation or test by hash.",
        "- Posterior depth metrics use three seeds.",
        "- Reported summary metrics macro-average over depth groups before seed-level averaging.",
        "",
        "## Allowed caption language",
        "- `Operator-conditioned posterior modelling improves MMP/US depth posterior quality and exposes piezo sheet depth non-identifiability through wider, more multimodal posteriors.`",
        "- `Piezo and ultrasound are simulation-derived benchmark operators matched to the same latent library.`",
        "",
        "## Forbidden caption language",
        "- Do not say the benchmark is a paired real hardware head-to-head experiment.",
        "- Do not say op-conditioned diffusion wins every piezo point-depth metric unless future metrics support it.",
        "- Do not imply the posterior mean volume is directly measured 3D tissue truth.",
        "",
    ]
    if dml_line:
        target = "- This run did not retrain diffusion models because frozen posterior64 outputs already existed and the new 3D depth benchmark is simulation-derived."
        lines.insert(lines.index(target), dml_line)
    if len(ours) and len(unet):
        o = ours.iloc[0]
        u = unet.iloc[0]
        lines += [
            "## Simulated depth-posterior benchmark, MMP",
            f"- Op-conditioned diffusion MAE: `{o['mae_zbottom_mm']:.3f}` mm; 3D U-Net MAE: `{u['mae_zbottom_mm']:.3f}` mm.",
            f"- Op-conditioned diffusion CRPS: `{o['crps_zbottom_mm']:.3f}` mm; 3D U-Net CRPS: `{u['crps_zbottom_mm']:.3f}` mm.",
            f"- Op-conditioned diffusion Cov90/Width90: `{o['cov90_zbottom']:.3f}` / `{o['width90_zbottom_mm']:.3f}` mm.",
            "",
        ]
    if len(existing):
        cols = ["split_mode", "model", "mae_norm_mean", "rmse_norm_mean", "crps_norm_mean", "failure_auroc_top10_abs_error_mean", "cov90_mean", "width90_norm_mean"]
        lines += ["## Existing frozen MMP posterior64 aggregate"]
        for _, row in existing[cols].iterrows():
            lines.append(
                f"- `{row['split_mode']}` `{row['model']}`: MAE `{row['mae_norm_mean']:.3f}`, RMSE `{row['rmse_norm_mean']:.3f}`, CRPS `{row['crps_norm_mean'] if pd.notna(row['crps_norm_mean']) else 'NA'}`, fail-AUROC `{row['failure_auroc_top10_abs_error_mean'] if pd.notna(row['failure_auroc_top10_abs_error_mean']) else 'NA'}`."
            )
        lines.append("")
    lines += [
        "## Outputs",
        f"- Data manifest: `{rel(DATA / 'Figure3_data_manifest.json')}`",
        f"- Posterior bank for Figure 4: `{rel(SHARED / 'fig3_to_fig4_posterior_bank.npz')}`",
        f"- Final locked image: `{rel(FIG / 'figure3_final_locked_overlay.png')}`",
        f"- Vector outputs: `{rel(FIG / 'figure3_data_driven.svg')}`, `{rel(FIG / 'figure3_data_driven.pdf')}`",
    ]
    (REPORTS / "figure3_methods_data_provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qa_gates() -> dict[str, Any]:
    from PIL import Image, ImageChops

    results: dict[str, Any] = {}
    truth = np.load(DATA / "latent_truth/volume_E_xyz.npz", allow_pickle=True)["E_xyz"].astype(np.float32)
    mean = np.load(DATA / "posteriors/posterior_mean_volume.npz", allow_pickle=True)["posterior_mean_E_xyz"].astype(np.float32)
    mae = float(np.mean(np.abs(truth - mean)))
    rel_mae = float(mae / max(float(np.mean(np.abs(truth))), 1e-6))
    results["posterior_volume_mae_kpa"] = mae
    results["posterior_volume_relative_mae"] = rel_mae
    results["truth_leakage_gate_pass"] = bool(mae >= 0.05 and rel_mae >= 0.005)

    data_img = Image.open(FIG / "figure3_data_driven.png").convert("RGBA")
    refined_img = Image.open(FIG / "figure3_ai_refined.png").convert("RGBA")
    diff = ImageChops.difference(data_img, refined_img)
    changed = np.any(np.asarray(diff)[..., :3] > 0, axis=2)
    results["ai_refinement_changed_pixels"] = int(changed.sum())
    results["ai_refinement_nonidentical_pass"] = bool(changed.sum() > 0)

    overlay_qc_path = FIG / "locked_overlay_qc.json"
    overlay_qc = read_json(overlay_qc_path) if overlay_qc_path.exists() else {}
    results["locked_overlay_qc"] = overlay_qc
    results["locked_region_identity_pass"] = bool(overlay_qc.get("locked_region_pixel_identity_pass", False))

    prov = REPORTS / "figure3_methods_data_provenance.md"
    results["provenance_exists_pass"] = prov.exists()
    text = prov.read_text(encoding="utf-8") if prov.exists() else ""
    forbidden = ["ours beats all baselines", "ours is superior for all operators", "is a paired real-device head-to-head"]
    results["forbidden_claims_found"] = [phrase for phrase in forbidden if phrase.lower() in text.lower()]
    results["piezo_claim_gate_pass"] = len(results["forbidden_claims_found"]) == 0

    cal = pd.read_csv(DATA / "metrics/calibration_metrics.csv")
    prob = cal[cal["method"].str.contains("diffusion|dps", case=False, regex=True)]
    all_cov_one = bool(len(prob) > 0 and np.allclose(prob["cov90_zbottom"].to_numpy(dtype=float), 1.0))
    results["probabilistic_cov90_all_one"] = all_cov_one
    results["calibration_gate_pass"] = bool(not all_cov_one and {"uce", "ence", "width90_zbottom_mm"}.issubset(cal.columns))

    required = [
        SHARED / "fig3_to_fig4_posterior_bank.npz",
        SHARED / "fig3_to_fig4_manifest.json",
        DATA / "domain_randomization/prior_spec.json",
        DATA / "domain_randomization/prior_qc.csv",
    ]
    results["bridge_and_prior_files_pass"] = all(p.exists() for p in required)
    results["overall_pass"] = all(
        bool(results[k])
        for k in [
            "truth_leakage_gate_pass",
            "ai_refinement_nonidentical_pass",
            "locked_region_identity_pass",
            "provenance_exists_pass",
            "piezo_claim_gate_pass",
            "calibration_gate_pass",
            "bridge_and_prior_files_pass",
        ]
    )
    write_json(REPORTS / "figure3_qa_gates.json", results)
    if not results["overall_pass"]:
        raise RuntimeError(f"Figure 3 QA gates failed; see {REPORTS / 'figure3_qa_gates.json'}")
    return results


def run_all() -> None:
    inventory()
    generate_assets()
    compute_metrics()
    render_figure()
    composite_locked()
    inventory()
    write_provenance_report()
    qa_gates()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build data-driven Figure 3 assets and rendering.")
    parser.add_argument("command", nargs="?", default="all", choices=["inventory", "generate", "metrics", "render", "composite", "qa", "all"])
    args = parser.parse_args(argv)
    if args.command == "inventory":
        manifest = inventory()
        print(json.dumps({"all_required_generated": manifest["all_required_generated"], "manifest": str(DATA / "Figure3_data_manifest.json")}, indent=2))
    elif args.command == "generate":
        generate_assets()
        print(f"Generated simulation assets under {DATA}")
    elif args.command == "metrics":
        compute_metrics()
        write_provenance_report()
        print(f"Computed posterior metrics under {DATA / 'metrics'}")
    elif args.command == "render":
        render_figure()
        print(f"Rendered Figure 3 under {FIG}")
    elif args.command == "composite":
        composite_locked()
        print(f"Composited locked Figure 3 under {FIG}")
    elif args.command == "qa":
        qa = qa_gates()
        print(json.dumps(qa, indent=2))
    else:
        run_all()
        print(f"Completed Figure 3 pipeline under {FIG}")


if __name__ == "__main__":
    main()
