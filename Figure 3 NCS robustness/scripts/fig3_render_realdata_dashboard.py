from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fig3_data_pipeline import BLUE, GREEN, ORANGE, PURPLE, RED, SEED, TEAL, forward_readout


DATA = ROOT / "Data" / "fig3"
FIG = ROOT / "figures" / "figure3"
REPORTS = ROOT / "reports"
MN_AUDIT = Path(
    r"H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_ncs_robustness_audit_20260531\fig3_ncs_robustness_audit_summary.json"
)

OUT_PNG = FIG / "figure3_actual_data_visualization.png"
OUT_PDF = FIG / "figure3_actual_data_visualization.pdf"
OUT_JSON = FIG / "figure3_actual_data_visualization_manifest.json"


METHODS = [
    ("direct_scalar_regression", "Direct reg.", "#8f8f8f"),
    ("unet_3d", "3D U-Net", "#3d7fc1"),
    ("kernel_gp", "Kernel/GP", "#6b55a3"),
    ("vanilla_3d_diffusion", "Vanilla diff.", "#34a853"),
    ("dps_inverse_diffusion", "DPS inverse", "#b552b7"),
    ("op_conditioned_diffusion_ours", "Op-cond. diff.", BLUE),
]

OPERATORS = [
    ("MMP", "MMP", BLUE),
    ("PIEZO", "Piezo", ORANGE),
    ("US", "Ultrasound", TEAL),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(BLUE)
        s.set_linewidth(0.9)
    ax.text(0.012, 0.985, letter, transform=ax.transAxes, ha="left", va="top", fontsize=15, fontweight="bold")
    ax.text(0.055, 0.982, title, transform=ax.transAxes, ha="left", va="top", fontsize=9.4, fontweight="bold")


def heat(ax: plt.Axes, arr: np.ndarray, title: str = "", cmap: str = "turbo", vmin: float | None = None, vmax: float | None = None) -> None:
    ax.imshow(arr, origin="lower", cmap=cmap, interpolation="bilinear", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7.2, pad=1.4)
    for s in ax.spines.values():
        s.set_linewidth(0.35)
        s.set_color("#b7c3db")


def posterior_samples(case_id: str, operator: str = "MMP", method: str = "op_conditioned_diffusion_ours") -> tuple[np.ndarray, float]:
    bank = np.load(DATA / "posteriors" / "zbottom_samples.npz", allow_pickle=True)
    idx = pd.read_csv(DATA / "posteriors" / "zbottom_samples_index.csv")
    mask = idx["case_id"].eq(case_id) & idx["operator"].eq(operator) & idx["method"].eq(method)
    row_idx = int(idx.index[mask][0])
    return bank["posterior_samples_zbottom"][row_idx], float(bank["true_zbottom_mm"][row_idx])


def plot_posterior(ax: plt.Axes, samples: np.ndarray, true_z: float, color: str = BLUE) -> None:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    ax.hist(samples, bins=26, density=True, color=color, alpha=0.24)
    if samples.size > 2:
        grid = np.linspace(0, 15, 300)
        try:
            dens = gaussian_kde(samples)(grid)
            ax.plot(grid, dens, color=color, lw=1.15)
        except Exception:
            pass
    ax.axvline(true_z, color=RED, lw=0.9)
    ax.set_xlim(0, 15)
    ax.set_yticks([])
    ax.tick_params(labelsize=5.1, length=2, pad=1)
    ax.spines[["top", "right", "left"]].set_visible(False)


def draw_status(ax: plt.Axes) -> None:
    audit = read_json(MN_AUDIT)
    manifest = read_json(DATA / "Figure3_data_manifest.json")
    paired = audit.get("paired_device", {})
    method2b = audit.get("method2b_512", {})
    rows = [
        ("Generated Figure 3 data assets", "PASS" if manifest.get("all_required_generated") else "CHECK", GREEN if manifest.get("all_required_generated") else ORANGE),
        ("Paired MMP/Piezo/US benchmark rows", f"{paired.get('all_three_complete_rows', 0)}/{paired.get('rows', 0)}", GREEN if paired.get("all_three_complete_rows", 0) == paired.get("rows", -1) else ORANGE),
        ("Method2B high-fidelity COMSOL queue", f"{method2b.get('complete_cases', 0)}/{method2b.get('planned_cases', 512)}", ORANGE),
        ("NCS robustness gate", audit.get("ncs_gate_status", "NOT_YET"), ORANGE if audit.get("ncs_gate_status") != "PASS" else GREEN),
    ]
    ax.axis("off")
    ax.text(0.01, 0.96, "Data status and guardrail", fontsize=9.5, fontweight="bold", color=BLUE, va="top")
    for i, (label, value, color) in enumerate(rows):
        y = 0.74 - i * 0.18
        ax.add_patch(mpl.patches.Rectangle((0.02, y - 0.055), 0.035, 0.07, fc="white", ec=color, lw=0.8, transform=ax.transAxes))
        ax.text(0.08, y, label, fontsize=7.1, va="center")
        ax.text(0.96, y, value, fontsize=7.1, va="center", ha="right", color=color, fontweight="bold")
    ax.text(
        0.02,
        0.06,
        "This dashboard is data-only: no V3 template data plots are used. Piezo/US are simulation-derived benchmark operators.",
        fontsize=6.9,
        color="#333333",
        va="bottom",
        wrap=True,
    )


def draw_projection_panel(fig: plt.Figure, spec) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "a", "Projection ambiguity from Data/fig3")
    sg = spec.subgridspec(3, 6, height_ratios=[0.18, 1.0, 0.72], width_ratios=[1.2, 1, 1, 1, 1, 0.12], hspace=0.26, wspace=0.17)
    eapp = np.load(DATA / "panel_a" / "eapp_common.npy")
    projs = np.load(DATA / "panel_a" / "projections.npy")
    states = np.load(DATA / "panel_a" / "alternative_3d_states.npz", allow_pickle=True)["E_xyz"]
    state_labels = ["shallow thin", "deep high-C", "thick intermediate", "heterogeneous core"]
    e_ax = fig.add_subplot(sg[1:, 0])
    im = e_ax.imshow(eapp, cmap="turbo", origin="lower", vmin=1, vmax=95, interpolation="bilinear")
    e_ax.set_xticks([])
    e_ax.set_yticks([])
    e_ax.set_title("Common apparent stiffness\n$E_{app}(x,y)$", fontsize=7.4)
    e_ax.text(0.02, -0.08, "5 mm", transform=e_ax.transAxes, fontsize=6)
    for i in range(4):
        s_ax = fig.add_subplot(sg[1, i + 1])
        heat(s_ax, states[i, :, 32, :], state_labels[i], vmin=1, vmax=130)
        p_ax = fig.add_subplot(sg[2, i + 1])
        heat(p_ax, projs[i], "projection", vmin=1, vmax=95)
    cax = fig.add_subplot(sg[1:, 5])
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.ax.set_title("kPa", fontsize=6, pad=3)


def draw_prior_panel(fig: plt.Figure, spec) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "b", "Domain-randomized prior distributions")
    sg = spec.subgridspec(3, 4, height_ratios=[0.18, 1, 1], hspace=0.38, wspace=0.30)
    domain = pd.read_csv(DATA / "domain_randomization" / "parameter_table.csv")
    cols = [
        ("z_bottom_mm", BLUE, "$z_{bottom}$ (mm)"),
        ("thickness_mm", GREEN, "thickness (mm)"),
        ("log10_contrast_x", PURPLE, "$log_{10}$ contrast"),
        ("log_layer_modulus_variation", ORANGE, "log layer variation"),
        ("readout_noise_rel", TEAL, "readout noise"),
        ("contact_pressure_kpa", "#9c6ade", "contact pressure"),
        ("lateral_psf_mm", "#c17c2c", "lateral PSF"),
        ("calibration_shift_rel", "#4a9f76", "calibration shift"),
    ]
    for i, (col, color, title) in enumerate(cols):
        h = fig.add_subplot(sg[i // 4 + 1, i % 4])
        vals = pd.to_numeric(domain[col], errors="coerce").dropna()
        h.hist(vals, bins=24, color=color, alpha=0.84, edgecolor="white", linewidth=0.25)
        h.set_title(title, fontsize=7.1)
        h.tick_params(labelsize=5.1, length=2, pad=1)
        h.spines[["top", "right"]].set_visible(False)


def draw_operator_panel(fig: plt.Figure, spec) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "c", "Operator readouts, depth kernels and lateral kernels")
    sg = spec.subgridspec(4, 3, height_ratios=[0.18, 1, 1, 1], hspace=0.42, wspace=0.30)
    readouts = np.load(DATA / "operators" / "operator_readouts_examples.npz", allow_pickle=True)
    lateral = pd.read_csv(DATA / "operators" / "lateral_psf.csv")
    kfiles = {"MMP": "mmp_kernel.npz", "PIEZO": "piezo_sheet_kernel.npz", "US": "ultrasound_strain_kernel.npz"}
    for row, (op, label, color) in enumerate(OPERATORS):
        rax = fig.add_subplot(sg[row + 1, 0])
        heat(rax, readouts[op], f"{label} readout")
        k = np.load(DATA / "operators" / kfiles[op])
        z = k["z_mm"]
        kd = k["Kd"]
        kax = fig.add_subplot(sg[row + 1, 1])
        kax.plot(kd / kd.max(), z, color=color, lw=1.5)
        kax.invert_yaxis()
        kax.set_xlabel("$K_d$ (norm.)", fontsize=6)
        kax.set_ylabel("z (mm)", fontsize=6)
        kax.tick_params(labelsize=5.1, length=2, pad=1)
        kax.spines[["top", "right"]].set_visible(False)
        lax = fig.add_subplot(sg[row + 1, 2])
        lax.plot(lateral["r_mm"], lateral[op], color=color, lw=1.5)
        lax.set_xlabel("r (mm)", fontsize=6)
        lax.set_ylabel("h(r)", fontsize=6)
        lax.tick_params(labelsize=5.1, length=2, pad=1)
        lax.spines[["top", "right"]].set_visible(False)


def draw_posterior_gallery(fig: plt.Figure, spec) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "d", "Representative posterior gallery from Data/fig3")
    truth = np.load(DATA / "latent_truth" / "volume_E_xyz.npz", allow_pickle=True)
    mean = np.load(DATA / "posteriors" / "posterior_mean_volume.npz", allow_pickle=True)["posterior_mean_E_xyz"]
    std = np.load(DATA / "posteriors" / "posterior_uncertainty_volume.npz", allow_pickle=True)["posterior_std_E_xyz"]
    cases = list(truth["case_id"])
    z = truth["z_mm"]
    meta = pd.read_csv(DATA / "metadata" / "cases.csv").set_index("case_id")
    show_cases = ["shallow_lesion", "mid_depth_lesion", "deep_lesion", "irregular_boundary", "low_contrast_ambiguous", "ood_deep"]
    sg = spec.subgridspec(len(show_cases) + 2, 6, height_ratios=[0.18, 0.16] + [1] * len(show_cases), hspace=0.20, wspace=0.18)
    headers = ["case", "readout", "true slice", "posterior mean", "uncertainty", "$p(z_{bottom})$"]
    for j, title in enumerate(headers):
        hax = fig.add_subplot(sg[1, j])
        hax.axis("off")
        hax.text(0.5, 0.35, title, ha="center", va="center", fontsize=7.1, fontweight="bold")
    for i, cid in enumerate(show_cases, start=2):
        ci = cases.index(cid)
        z_mid = 0.5 * (float(meta.loc[cid, "z_top_mm"]) + float(meta.loc[cid, "z_bottom_mm"]))
        zidx = int(np.argmin(np.abs(z - z_mid)))
        label_ax = fig.add_subplot(sg[i, 0])
        label_ax.axis("off")
        label_ax.text(0.02, 0.55, meta.loc[cid, "label"].replace(" ", "\n"), ha="left", va="center", fontsize=6.1)
        readout = forward_readout(truth["E_xyz"][ci], "MMP")
        arrays = [
            (readout, "turbo", None, None),
            (truth["E_xyz"][ci, zidx], "turbo", 1, 120),
            (mean[ci, zidx], "turbo", 1, 120),
            (std[ci, zidx], "magma", None, None),
        ]
        for j, (arr, cmap, vmin, vmax) in enumerate(arrays, start=1):
            h = fig.add_subplot(sg[i, j])
            heat(h, arr, "", cmap=cmap, vmin=vmin, vmax=vmax)
        pax = fig.add_subplot(sg[i, 5])
        samples, true_z = posterior_samples(cid)
        plot_posterior(pax, samples, true_z)
        if i == len(show_cases) + 1:
            pax.set_xlabel("$z_{bottom}$ (mm)", fontsize=6)


def draw_metric_panel(fig: plt.Figure, spec) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "e", "Posterior metrics from Data/fig3")
    metrics = pd.read_csv(DATA / "metrics" / "depth_metrics.csv")
    rc = pd.read_csv(DATA / "metrics" / "risk_coverage.csv")
    ex = pd.read_csv(DATA / "metrics" / "error_vs_extrapolation.csv")
    mm = pd.read_csv(DATA / "metrics" / "multimodality_metrics.csv")
    sg = spec.subgridspec(3, 3, height_ratios=[0.17, 1, 1], hspace=0.42, wspace=0.32)
    mmp = metrics[metrics["operator"].eq("MMP")].set_index("method")

    ax1 = fig.add_subplot(sg[1, 0])
    x = np.arange(len(METHODS))
    ax1.bar(x - 0.18, [mmp.loc[m, "mae_zbottom_mm"] for m, _, _ in METHODS], width=0.34, color=[c for _, _, c in METHODS], alpha=0.95, label="MAE")
    ax1.bar(x + 0.18, [mmp.loc[m, "crps_zbottom_mm"] for m, _, _ in METHODS], width=0.34, color=[c for _, _, c in METHODS], alpha=0.40, label="CRPS")
    ax1.set_title("MMP depth error / proper score", fontsize=7.2)
    ax1.set_ylabel("mm", fontsize=6)
    ax1.set_xticks(x, [lab.replace(" ", "\n") for _, lab, _ in METHODS], fontsize=5.0)
    ax1.legend(fontsize=5.2, frameon=False)
    ax1.tick_params(labelsize=5.0, length=2, pad=1)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(sg[1, 1])
    for method, label, color in METHODS:
        row = mmp.loc[method]
        ax2.scatter(row["width90_zbottom_mm"], abs(row["cov90_zbottom"] - 0.90), s=32, color=color, edgecolor="white", linewidth=0.4, label=label)
    ax2.set_title("Calibration / sharpness", fontsize=7.2)
    ax2.set_xlabel("Width90 (mm)", fontsize=6)
    ax2.set_ylabel("|Cov90 - 0.90|", fontsize=6)
    ax2.tick_params(labelsize=5.0, length=2, pad=1)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(sg[1, 2])
    ours = metrics[metrics["method"].eq("op_conditioned_diffusion_ours")].set_index("operator").reindex(["MMP", "PIEZO", "US"])
    freq = mm[mm["method"].eq("op_conditioned_diffusion_ours")].groupby("operator")["multimodal_flag"].mean().reindex(["MMP", "PIEZO", "US"])
    xo = np.arange(3)
    ax3.bar(xo - 0.17, ours["width90_zbottom_mm"], width=0.32, color=[BLUE, ORANGE, TEAL], alpha=0.55, label="Width90")
    ax3b = ax3.twinx()
    ax3b.bar(xo + 0.17, freq.values, width=0.32, color=[BLUE, ORANGE, TEAL], alpha=0.95, label="multimodal")
    ax3.set_xticks(xo, ["MMP", "Piezo", "US"], fontsize=5.5)
    ax3.set_ylabel("Width90 (mm)", fontsize=6)
    ax3b.set_ylabel("multimodal freq.", fontsize=6)
    ax3.set_title("Operator observability", fontsize=7.2)
    ax3.tick_params(labelsize=5.0, length=2, pad=1)
    ax3b.tick_params(labelsize=5.0, length=2, pad=1)
    ax3.spines[["top"]].set_visible(False)
    ax3b.spines[["top"]].set_visible(False)

    ax4 = fig.add_subplot(sg[2, 0])
    for method, label, color in [METHODS[1], METHODS[3], METHODS[5]]:
        sub = rc[(rc["operator"].eq("MMP")) & rc["method"].eq(method)]
        sub = sub.groupby("coverage_fraction", as_index=False)["risk_mae_mm"].mean()
        ax4.plot(sub["coverage_fraction"], sub["risk_mae_mm"], color=color, lw=1.4, label=label)
    ax4.set_title("Risk coverage, MMP", fontsize=7.2)
    ax4.set_xlabel("retained fraction", fontsize=6)
    ax4.set_ylabel("risk MAE (mm)", fontsize=6)
    ax4.legend(fontsize=5.2, frameon=False)
    ax4.tick_params(labelsize=5.0, length=2, pad=1)
    ax4.spines[["top", "right"]].set_visible(False)

    ax5 = fig.add_subplot(sg[2, 1])
    for op, label, color in OPERATORS:
        sub = ex[(ex["operator"].eq(op)) & ex["method"].eq("op_conditioned_diffusion_ours")]
        bins = np.linspace(0, sub["d_out_mm"].max() + 1e-6, 7)
        centers, vals = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            ss = sub[(sub["d_out_mm"] >= lo) & (sub["d_out_mm"] <= hi)]
            if len(ss):
                centers.append((lo + hi) / 2)
                vals.append(ss["absolute_error_mm"].mean())
        ax5.plot(centers, vals, marker="o", ms=2.8, lw=1.3, color=color, label=label)
    ax5.set_title("Error vs extrapolation", fontsize=7.2)
    ax5.set_xlabel("distance outside train support (mm)", fontsize=6)
    ax5.set_ylabel("MAE (mm)", fontsize=6)
    ax5.legend(fontsize=5.2, frameon=False)
    ax5.tick_params(labelsize=5.0, length=2, pad=1)
    ax5.spines[["top", "right"]].set_visible(False)

    ax6 = fig.add_subplot(sg[2, 2])
    depth_group = pd.read_csv(DATA / "posteriors" / "zbottom_samples_index.csv")
    bank = np.load(DATA / "posteriors" / "zbottom_samples.npz", allow_pickle=True)
    summary = depth_group.copy()
    summary["posterior_mean"] = bank["posterior_samples_zbottom"].mean(axis=1)
    summary["abs_error"] = np.abs(summary["posterior_mean"] - summary["true_zbottom_mm"])
    sub = summary[summary["method"].eq("op_conditioned_diffusion_ours")]
    grouped = sub.groupby(["operator", "depth_group"], as_index=False)["abs_error"].mean()
    order = ["shallow", "mid", "deep", "ood"]
    for op, label, color in OPERATORS:
        vals = grouped[grouped["operator"].eq(op)].set_index("depth_group").reindex(order)["abs_error"]
        ax6.plot(order, vals, marker="o", lw=1.3, color=color, label=label)
    ax6.set_title("Depth-group MAE", fontsize=7.2)
    ax6.set_ylabel("MAE (mm)", fontsize=6)
    ax6.tick_params(labelsize=5.0, length=2, pad=1)
    ax6.legend(fontsize=5.2, frameon=False)
    ax6.spines[["top", "right"]].set_visible(False)


def draw_source_panel(fig: plt.Figure, spec) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "f", "Data sources rendered in this standalone file")
    ax.axis("off")
    lines = [
        ("Panel a", "Data/fig3/panel_a/eapp_common.npy; projections.npy; alternative_3d_states.npz"),
        ("Panel b", "Data/fig3/domain_randomization/parameter_table.csv"),
        ("Panel c", "Data/fig3/operators/*kernel.npz; lateral_psf.csv; operator_readouts_examples.npz"),
        ("Panel d", "Data/fig3/latent_truth; Data/fig3/posteriors/posterior_mean/uncertainty/zbottom_samples"),
        ("Panel e", "Data/fig3/metrics/depth_metrics.csv; calibration/risk/extrapolation/multimodality metrics"),
        ("Source label", "SIMULATED_BENCHMARK, with missing paired real hardware explicitly not synthesized"),
    ]
    y = 0.80
    for key, value in lines:
        ax.text(0.04, y, key, fontsize=7.2, fontweight="bold", color=BLUE, va="top")
        ax.text(0.22, y, value, fontsize=6.4, va="top", wrap=True)
        y -= 0.13


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.55,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(24, 16), dpi=300)
    fig.suptitle(
        "Figure 3 actual-data visualization | Data/fig3 only, no V3 AI data plots",
        fontsize=15,
        fontweight="bold",
        y=0.987,
    )
    outer = fig.add_gridspec(
        3,
        3,
        left=0.018,
        right=0.988,
        top=0.945,
        bottom=0.035,
        width_ratios=[1.08, 1.40, 0.95],
        height_ratios=[1.00, 1.42, 1.35],
        hspace=0.20,
        wspace=0.065,
    )
    draw_projection_panel(fig, outer[0, 0])
    draw_prior_panel(fig, outer[0, 1])
    status_ax = fig.add_subplot(outer[0, 2])
    draw_status(status_ax)
    draw_operator_panel(fig, outer[1, 0])
    draw_posterior_gallery(fig, outer[1, 1:])
    draw_metric_panel(fig, outer[2, :2])
    draw_source_panel(fig, outer[2, 2])
    fig.text(
        0.5,
        0.012,
        "Rendered directly from refreshed Data/fig3 arrays/tables and robustness-audit summaries; piezo/ultrasound are simulation-derived benchmark operators.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=BLUE,
        fontstyle="italic",
        fontweight="bold",
    )
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_png": str(OUT_PNG),
        "output_pdf": str(OUT_PDF),
        "png_sha256": sha256(OUT_PNG),
        "data_root": str(DATA),
        "source": "Data/fig3 refreshed by scripts/fig3_data_pipeline.py all",
        "no_v3_template_used": True,
        "panels": {
            "a": ["panel_a/eapp_common.npy", "panel_a/projections.npy", "panel_a/alternative_3d_states.npz"],
            "b": ["domain_randomization/parameter_table.csv"],
            "c": ["operators/*kernel.npz", "operators/lateral_psf.csv", "operators/operator_readouts_examples.npz"],
            "d": ["latent_truth/volume_E_xyz.npz", "posteriors/posterior_mean_volume.npz", "posteriors/posterior_uncertainty_volume.npz", "posteriors/zbottom_samples.npz"],
            "e": ["metrics/depth_metrics.csv", "metrics/risk_coverage.csv", "metrics/error_vs_extrapolation.csv", "metrics/multimodality_metrics.csv"],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
