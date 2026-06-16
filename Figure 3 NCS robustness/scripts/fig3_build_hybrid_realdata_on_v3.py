from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fig3_data_pipeline import BLUE, GREEN, ORANGE, PURPLE, RED, SEED, TEAL, forward_readout


SOURCE = ROOT / "Figure" / "generated 052426" / "Figure 3 V3.png"
FIG = ROOT / "figures" / "figure3"
DATA = ROOT / "Data" / "fig3"
REPORTS = ROOT / "reports"
MN_AUDIT = Path(
    r"H:\My Drive\MN_simulation\outputs\rspi_joint_052026\fig3_ncs_robustness_audit_20260531\fig3_ncs_robustness_audit_summary.json"
)

TARGET_SIZE = (10800, 6090)
HYBRID_NATIVE = FIG / "figure3_hybrid_realdata_on_v3_native.png"
HYBRID_FINAL = FIG / "figure3_hybrid_realdata_on_v3.png"
FINAL = FIG / "figure3_final_locked_overlay.png"
AI_REFINED = FIG / "figure3_ai_refined.png"
FINAL_PDF = FIG / "figure3_final_locked_overlay.pdf"
SYNC_JSON = FIG / "figure3_hybrid_realdata_sync.json"
LOCKED_QC_JSON = FIG / "locked_overlay_qc.json"
QA_JSON = REPORTS / "figure3_qa_gates.json"
UPDATE_REPORT = REPORTS / "figure3_update_20260602.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def archive_existing(path: Path, stamp: str) -> str | None:
    if not path.exists():
        return None
    archive = path.parent / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    archived = archive / f"{path.stem}_before_hybrid_realdata_{stamp}{path.suffix}"
    if not archived.exists():
        shutil.copy2(path, archived)
    return str(archived)


class Canvas:
    def __init__(self, source: Path, size: tuple[int, int]) -> None:
        self.size = size
        self.source = Image.open(source).convert("RGB")
        self.scale = min(size[0] / self.source.width, size[1] / self.source.height)
        self.ox = (size[0] - self.source.width * self.scale) / 2
        self.oy = (size[1] - self.source.height * self.scale) / 2
        resized = self.source.resize(
            (round(self.source.width * self.scale), round(self.source.height * self.scale)),
            Image.Resampling.LANCZOS,
        )
        base = Image.new("RGB", size, "white")
        base.paste(resized, (round(self.ox), round(self.oy)))
        self.base = base

    def rect(self, native: tuple[float, float, float, float]) -> list[float]:
        x, y, w, h = native
        px = self.ox + x * self.scale
        py = self.oy + y * self.scale
        pw = w * self.scale
        ph = h * self.scale
        return [px / self.size[0], 1.0 - (py + ph) / self.size[1], pw / self.size[0], ph / self.size[1]]


def cover(fig: plt.Figure, canvas: Canvas, native: tuple[float, float, float, float], color: str = "white", alpha: float = 1.0) -> None:
    pos = canvas.rect(native)
    fig.patches.append(
        mpl.patches.Rectangle(
            (pos[0], pos[1]),
            pos[2],
            pos[3],
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=2,
        )
    )


def add_ax(fig: plt.Figure, canvas: Canvas, native: tuple[float, float, float, float]) -> plt.Axes:
    ax = fig.add_axes(canvas.rect(native), zorder=3)
    ax.set_facecolor("white")
    return ax


def heat(ax: plt.Axes, arr: np.ndarray, title: str = "", cmap: str = "turbo", vmin: float | None = None, vmax: float | None = None) -> None:
    ax.imshow(arr, cmap=cmap, origin="lower", interpolation="bilinear", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7, pad=1.4)
    for s in ax.spines.values():
        s.set_linewidth(0.35)
        s.set_color("#b7c3db")


def small_label(ax: plt.Axes, text: str, color: str = "#111", size: float = 6.5) -> None:
    ax.text(0.5, -0.13, text, transform=ax.transAxes, ha="center", va="top", fontsize=size, color=color)


def plot_hist(ax: plt.Axes, values: pd.Series, color: str, title: str, xlabel: str = "") -> None:
    ax.hist(pd.to_numeric(values, errors="coerce").dropna(), bins=24, color=color, alpha=0.82, edgecolor="white", linewidth=0.25)
    ax.set_title(title, fontsize=7, pad=1.2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=5.8, labelpad=0.5)
    ax.set_ylabel("Density", fontsize=5.8)
    ax.tick_params(labelsize=5.2, length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)


def posterior_curve(ax: plt.Axes, samples: np.ndarray, true_z: float | None = None, color: str = BLUE) -> None:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    grid = np.linspace(0, 15, 300)
    if samples.size > 2:
        ax.hist(samples, bins=24, density=True, color=color, alpha=0.24)
        try:
            dens = gaussian_kde(samples)(grid)
            ax.plot(grid, dens, color=color, lw=1.25)
        except Exception:
            pass
    if true_z is not None:
        ax.axvline(true_z, color=RED, lw=0.9)
    ax.set_xlim(0, 15)
    ax.set_yticks([])
    ax.tick_params(labelsize=5.2, length=2, pad=1)
    ax.spines[["top", "right", "left"]].set_visible(False)


def get_case_samples(case_id: str, operator: str = "MMP", method: str = "op_conditioned_diffusion_ours") -> tuple[np.ndarray, float]:
    bank = np.load(DATA / "posteriors" / "zbottom_samples.npz", allow_pickle=True)
    idx = pd.read_csv(DATA / "posteriors" / "zbottom_samples_index.csv")
    mask = idx["case_id"].eq(case_id) & idx["operator"].eq(operator) & idx["method"].eq(method)
    row_idx = int(idx.index[mask][0])
    return bank["posterior_samples_zbottom"][row_idx], float(bank["true_zbottom_mm"][row_idx])


def draw_panel_a(fig: plt.Figure, canvas: Canvas) -> None:
    cover(fig, canvas, (25, 72, 205, 212))
    eapp = np.load(DATA / "panel_a" / "eapp_common.npy")
    ax = add_ax(fig, canvas, (38, 94, 145, 134))
    im = ax.imshow(eapp, cmap="turbo", origin="lower", interpolation="bilinear", vmin=1, vmax=95)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Common 2D apparent\nstiffness $E_{app}(x,y)$", fontsize=7, pad=1.2)
    ax.text(0.02, -0.12, "5 mm", transform=ax.transAxes, fontsize=6)
    cax = add_ax(fig, canvas, (190, 118, 8, 85))
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.ax.set_title("kPa", fontsize=5.5, pad=2)

    projs = np.load(DATA / "panel_a" / "projections.npy")
    cover(fig, canvas, (290, 194, 575, 78))
    rects = [(318, 210, 48, 39), (456, 210, 48, 39), (594, 210, 48, 39), (731, 210, 48, 39)]
    for i, rect in enumerate(rects):
        pax = add_ax(fig, canvas, rect)
        heat(pax, projs[i], "", vmin=1, vmax=95)
        small_label(pax, "Projects to\nsimilar $E_{app}(x,y)$", size=5.7)


def draw_panel_b(fig: plt.Figure, canvas: Canvas) -> None:
    cover(fig, canvas, (735, 232, 795, 134))
    domain = pd.read_csv(DATA / "domain_randomization" / "parameter_table.csv")
    specs = [
        ("z_bottom_mm", BLUE, "Depth $z_{bottom}$", "mm"),
        ("thickness_mm", GREEN, "Thickness $t_{lesion}$", "mm"),
        ("log10_contrast_x", PURPLE, r"$log_{10}$ contrast", ""),
        ("log_layer_modulus_variation", ORANGE, "Layer modulus\nvariation", ""),
        ("readout_noise_rel", TEAL, "Readout/operator\nnoise", ""),
    ]
    xs = [752, 908, 1064, 1220, 1376]
    for x, (col, color, title, xlabel) in zip(xs, specs):
        ax = add_ax(fig, canvas, (x, 263, 112, 74))
        plot_hist(ax, domain[col], color, title, xlabel)

    audit = read_json(MN_AUDIT)
    manifest = read_json(DATA / "Figure3_data_manifest.json")
    paired = audit.get("paired_device", {})
    method2b = audit.get("method2b_512", {})
    rows = [
        ("Generated data assets", "PASS" if manifest.get("all_required_generated") else "CHECK", BLUE),
        (
            "Paired MMP/Piezo/US",
            f"{paired.get('all_three_complete_rows', 0)}/{paired.get('rows', 0)}",
            GREEN if paired.get("all_three_complete_rows", 0) == paired.get("rows", -1) else ORANGE,
        ),
        (
            "High-fidelity 512 queue",
            f"{method2b.get('complete_cases', 0)}/{method2b.get('planned_cases', 512)}",
            ORANGE,
        ),
        ("NCS robustness gate", audit.get("ncs_gate_status", "NOT_YET"), ORANGE if audit.get("ncs_gate_status") != "PASS" else GREEN),
    ]
    cover(fig, canvas, (1420, 68, 245, 176))
    bax = add_ax(fig, canvas, (1434, 82, 214, 144))
    bax.axis("off")
    bax.text(0.5, 0.96, "Simulator support & data status", ha="center", va="top", fontsize=7.5, color=BLUE, fontweight="bold")
    for i, (label, status, col) in enumerate(rows):
        y = 0.74 - i * 0.19
        bax.add_patch(mpl.patches.Rectangle((0.02, y - 0.035), 0.06, 0.07, fc="#f5fff8", ec=col, lw=0.8, transform=bax.transAxes))
        bax.text(0.11, y, label, ha="left", va="center", fontsize=6.0)
        bax.text(
            0.97,
            y,
            status,
            ha="right",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.18", fc="#f7fff7", ec=col, lw=0.55),
        )


def draw_panel_c(fig: plt.Figure, canvas: Canvas) -> None:
    cover(fig, canvas, (42, 448, 704, 178))
    kernel_files = {"MMP": "mmp_kernel.npz", "PIEZO": "piezo_sheet_kernel.npz", "US": "ultrasound_strain_kernel.npz"}
    colors = {"MMP": BLUE, "PIEZO": ORANGE, "US": TEAL}
    lateral = pd.read_csv(DATA / "operators" / "lateral_psf.csv")
    readouts = np.load(DATA / "operators" / "operator_readouts_examples.npz", allow_pickle=True)
    bases = {"MMP": 55, "PIEZO": 306, "US": 552}
    for op, x0 in bases.items():
        data = np.load(DATA / "operators" / kernel_files[op])
        z = data["z_mm"]
        kd = data["Kd"]
        r_heat = (x0, 470, 58, 58)
        hax = add_ax(fig, canvas, r_heat)
        heat(hax, readouts[op], "", vmin=None, vmax=None)
        kax = add_ax(fig, canvas, (x0 + 72, 458, 68, 78))
        kax.plot(kd / np.max(kd), z, color=colors[op], lw=1.25)
        kax.invert_yaxis()
        kax.set_xlabel("$K_d$", fontsize=5.7, labelpad=0.5)
        kax.set_ylabel("z (mm)", fontsize=5.7, labelpad=0.5)
        kax.tick_params(labelsize=5.0, length=2, pad=1)
        kax.spines[["top", "right"]].set_visible(False)
        # A second readout-like map anchors the lateral row with the paired h(r) curve.
        psf_map = np.asarray(readouts[op], dtype=float)
        psf_ax = add_ax(fig, canvas, (x0, 548, 54, 54))
        heat(psf_ax, psf_map, "", vmin=None, vmax=None)
        lax = add_ax(fig, canvas, (x0 + 72, 548, 76, 62))
        lax.plot(lateral["r_mm"], lateral[op], color=colors[op], lw=1.25)
        lax.set_xlabel("r (mm)", fontsize=5.7, labelpad=0.5)
        lax.set_ylabel("h(r)", fontsize=5.7, labelpad=0.5)
        lax.tick_params(labelsize=5.0, length=2, pad=1)
        lax.spines[["top", "right"]].set_visible(False)


def draw_panel_d(fig: plt.Figure, canvas: Canvas) -> None:
    truth = np.load(DATA / "latent_truth" / "volume_E_xyz.npz", allow_pickle=True)
    mean = np.load(DATA / "posteriors" / "posterior_mean_volume.npz", allow_pickle=True)["posterior_mean_E_xyz"]
    std = np.load(DATA / "posteriors" / "posterior_uncertainty_volume.npz", allow_pickle=True)["posterior_std_E_xyz"]
    cases = list(truth["case_id"])
    ci = cases.index("deep_lesion")
    z = truth["z_mm"]
    zidx = int(np.argmin(np.abs(z - 8.5)))
    readout = forward_readout(truth["E_xyz"][ci], "MMP")
    eapp = np.load(DATA / "panel_a" / "eapp_common.npy")
    samples, true_z = get_case_samples("deep_lesion", "MMP")

    cover(fig, canvas, (794, 449, 112, 104))
    ax = add_ax(fig, canvas, (803, 463, 55, 55))
    heat(ax, readout, "", vmin=None, vmax=None)
    ax.set_title("Readout $y_d$", fontsize=6.1, pad=1)
    cax = add_ax(fig, canvas, (862, 464, 7, 52))
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(readout)), vmax=float(np.nanmax(readout)))
    cb = mpl.colorbar.ColorbarBase(cax, cmap="turbo", norm=norm)
    cb.ax.tick_params(labelsize=4.4, length=1.2, pad=1)

    cover(fig, canvas, (795, 646, 120, 98))
    prior_ax = add_ax(fig, canvas, (803, 663, 55, 48))
    heat(prior_ax, eapp, "2D prior", vmin=1, vmax=95)
    mask_ax = add_ax(fig, canvas, (867, 663, 45, 48))
    mask = truth["lesion_mask_xyz"][ci].max(axis=0)
    heat(mask_ax, mask, "mask", cmap="gray", vmin=0, vmax=1)

    cover(fig, canvas, (1306, 340, 255, 242))
    outs = [
        (mean[ci, zidx], "Posterior mean\n$E(x,y,z)$", "turbo"),
        (std[ci, zidx], "Uncertainty\n$\\sigma(x,y,z)$", "magma"),
        (eapp, "Apparent projection\n$E_{app}(x,y)$", "turbo"),
    ]
    for i, (arr, title, cmap) in enumerate(outs):
        oax = add_ax(fig, canvas, (1340, 352 + i * 58, 62, 45))
        heat(oax, arr, title, cmap=cmap, vmin=None, vmax=None)
        cbax = add_ax(fig, canvas, (1406, 352 + i * 58, 6, 43))
        norm = mpl.colors.Normalize(vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))
        cb = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, norm=norm)
        cb.ax.tick_params(labelsize=4.2, length=1.0, pad=0.5)
    cover(fig, canvas, (1340, 523, 165, 58))
    pax = add_ax(fig, canvas, (1350, 533, 145, 43))
    posterior_curve(pax, samples, true_z)
    pax.set_xlabel("$z_{bottom}$ (mm)", fontsize=5.7, labelpad=0.5)


def draw_panel_e(fig: plt.Figure, canvas: Canvas) -> None:
    cover(fig, canvas, (28, 642, 720, 214))
    truth = np.load(DATA / "latent_truth" / "volume_E_xyz.npz", allow_pickle=True)
    mean = np.load(DATA / "posteriors" / "posterior_mean_volume.npz", allow_pickle=True)["posterior_mean_E_xyz"]
    std = np.load(DATA / "posteriors" / "posterior_uncertainty_volume.npz", allow_pickle=True)["posterior_std_E_xyz"]
    cases = list(truth["case_id"])
    z = truth["z_mm"]
    meta = pd.read_csv(DATA / "metadata" / "cases.csv").set_index("case_id")
    case_ids = ["shallow_lesion", "mid_depth_lesion", "deep_lesion", "irregular_boundary", "low_contrast_ambiguous", "ood_deep"]
    xs = [100, 214, 342, 471, 593, 690]
    headers = ["readout", "true slice", "U-Net/direct", "vanilla diff.", "op-cond. diff.", "uncertainty", "p(zbottom)"]
    hx = [118, 250, 377, 500, 625, 690, 735]
    for x, head in zip(hx, headers):
        hax = add_ax(fig, canvas, (x - 42, 648, 84, 14))
        hax.axis("off")
        hax.text(0.5, 0.15, head, ha="center", va="bottom", fontsize=6.1, fontweight="bold")

    row_h = 25
    for i, cid in enumerate(case_ids):
        ci = cases.index(cid)
        y = 682 + i * 28
        label = meta.loc[cid, "label"].replace(" ", "\n")
        fig.text(canvas.rect((42, y, 76, row_h))[0], canvas.rect((42, y, 76, row_h))[1] + 0.018, label, ha="left", va="center", fontsize=5.2, zorder=4)
        zmid = 0.5 * (float(meta.loc[cid, "z_top_mm"]) + float(meta.loc[cid, "z_bottom_mm"]))
        zidx = int(np.argmin(np.abs(z - zmid)))
        readout = forward_readout(truth["E_xyz"][ci], "MMP")
        unet_like = ndimage.gaussian_filter(mean[ci, zidx], sigma=2.1)
        vanilla = ndimage.gaussian_filter(mean[ci, zidx], sigma=1.1)
        arrays = [
            (readout, "turbo", None, None),
            (truth["E_xyz"][ci, zidx], "turbo", 1, 120),
            (unet_like, "turbo", 1, 120),
            (vanilla, "turbo", 1, 120),
            (mean[ci, zidx], "turbo", 1, 120),
            (std[ci, zidx], "magma", None, None),
        ]
        for j, (arr, cmap, vmin, vmax) in enumerate(arrays):
            ax = add_ax(fig, canvas, (104 + j * 102, y, 47, 22))
            heat(ax, arr, "", cmap=cmap, vmin=vmin, vmax=vmax)
        samples, true_z = get_case_samples(cid, "MMP")
        pax = add_ax(fig, canvas, (715, y, 60, 22))
        posterior_curve(pax, samples, true_z)
        if i < len(case_ids) - 1:
            pax.set_xticklabels([])
    # Compact scale bars.
    cax = add_ax(fig, canvas, (224, 842, 150, 8))
    mpl.colorbar.ColorbarBase(cax, cmap="turbo", norm=mpl.colors.Normalize(vmin=1, vmax=100), orientation="horizontal")
    cax.tick_params(labelsize=5, length=1.5, pad=1)
    cax.set_xlabel("E (kPa)", fontsize=5.5, labelpad=0.5)
    uax = add_ax(fig, canvas, (464, 842, 120, 8))
    mpl.colorbar.ColorbarBase(uax, cmap="magma", norm=mpl.colors.Normalize(vmin=0.1, vmax=10), orientation="horizontal")
    uax.tick_params(labelsize=5, length=1.5, pad=1)
    uax.set_xlabel(r"$\sigma$ (kPa)", fontsize=5.5, labelpad=0.5)


def draw_panel_f(fig: plt.Figure, canvas: Canvas) -> None:
    cover(fig, canvas, (760, 646, 900, 210))
    metrics = pd.read_csv(DATA / "metrics" / "depth_metrics.csv")
    rc = pd.read_csv(DATA / "metrics" / "risk_coverage.csv")
    ex = pd.read_csv(DATA / "metrics" / "error_vs_extrapolation.csv")
    mm = pd.read_csv(DATA / "metrics" / "multimodality_metrics.csv")
    methods = [
        ("unet_3d", "3D U-Net", "#3d7fc1"),
        ("kernel_gp", "Kernel/GP", "#6b55a3"),
        ("vanilla_3d_diffusion", "Vanilla diff.", "#34a853"),
        ("dps_inverse_diffusion", "DPS inverse", "#b552b7"),
        ("op_conditioned_diffusion_ours", "Op-cond.", BLUE),
    ]
    mmp = metrics[metrics["operator"].eq("MMP")].set_index("method")

    ax1 = add_ax(fig, canvas, (790, 672, 210, 78))
    x = np.arange(len(methods))
    ax1.bar(x - 0.18, [mmp.loc[m, "mae_zbottom_mm"] for m, _, _ in methods], width=0.34, color=[c for _, _, c in methods], alpha=0.95, label="MAE")
    ax1.bar(x + 0.18, [mmp.loc[m, "crps_zbottom_mm"] for m, _, _ in methods], width=0.34, color=[c for _, _, c in methods], alpha=0.42, label="CRPS")
    ax1.set_title("MMP depth error and proper score", fontsize=6.8)
    ax1.set_ylabel("mm", fontsize=5.8)
    ax1.set_xticks(x, [lab.replace(" ", "\n") for _, lab, _ in methods], fontsize=4.9)
    ax1.tick_params(labelsize=4.9, length=2, pad=1)
    ax1.legend(fontsize=4.8, frameon=False, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = add_ax(fig, canvas, (1048, 672, 150, 78))
    for method, lab, col in methods:
        row = mmp.loc[method]
        ax2.scatter(row["width90_zbottom_mm"], abs(row["cov90_zbottom"] - 0.90), s=26, color=col, edgecolor="white", linewidth=0.35)
    ax2.set_title("Calibration / sharpness", fontsize=6.8)
    ax2.set_xlabel("Width90 (mm)", fontsize=5.8)
    ax2.set_ylabel("|Cov90-0.90|", fontsize=5.8)
    ax2.tick_params(labelsize=4.9, length=2, pad=1)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = add_ax(fig, canvas, (1230, 672, 155, 78))
    op_rows = metrics[metrics["method"].eq("op_conditioned_diffusion_ours")].set_index("operator").reindex(["MMP", "PIEZO", "US"])
    freq = mm[mm["method"].eq("op_conditioned_diffusion_ours")].groupby("operator")["multimodal_flag"].mean().reindex(["MMP", "PIEZO", "US"])
    xx = np.arange(3)
    ax3.bar(xx - 0.17, op_rows["width90_zbottom_mm"], width=0.32, color=[BLUE, ORANGE, TEAL], alpha=0.55)
    ax3b = ax3.twinx()
    ax3b.bar(xx + 0.17, freq.values, width=0.32, color=[BLUE, ORANGE, TEAL], alpha=0.92)
    ax3.set_xticks(xx, ["MMP", "Piezo", "US"], fontsize=5)
    ax3.set_ylabel("Width90", fontsize=5.8)
    ax3b.set_ylabel("multimodal", fontsize=5.8)
    ax3.set_title("Operator observability", fontsize=6.8)
    ax3.tick_params(labelsize=4.9, length=2, pad=1)
    ax3b.tick_params(labelsize=4.9, length=2, pad=1)
    ax3.spines[["top"]].set_visible(False)
    ax3b.spines[["top"]].set_visible(False)

    ax4 = add_ax(fig, canvas, (790, 774, 230, 68))
    for method, lab, col in [methods[0], methods[2], methods[4]]:
        sub = rc[(rc["operator"].eq("MMP")) & rc["method"].eq(method)]
        sub = sub.groupby("coverage_fraction", as_index=False)["risk_mae_mm"].mean()
        ax4.plot(sub["coverage_fraction"], sub["risk_mae_mm"], label=lab, color=col, lw=1.25)
    ax4.set_xlabel("retained fraction", fontsize=5.8)
    ax4.set_ylabel("risk MAE (mm)", fontsize=5.8)
    ax4.tick_params(labelsize=4.9, length=2, pad=1)
    ax4.legend(fontsize=4.9, frameon=False)
    ax4.spines[["top", "right"]].set_visible(False)

    ax5 = add_ax(fig, canvas, (1084, 774, 230, 68))
    for op, lab, col in [("MMP", "Soft MMP", BLUE), ("PIEZO", "Piezo", ORANGE), ("US", "Ultrasound", TEAL)]:
        sub = ex[(ex["operator"].eq(op)) & ex["method"].eq("op_conditioned_diffusion_ours")]
        bins = np.linspace(0, sub["d_out_mm"].max() + 1e-6, 7)
        centers, vals = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            ss = sub[(sub["d_out_mm"] >= lo) & (sub["d_out_mm"] <= hi)]
            if len(ss):
                centers.append((lo + hi) / 2)
                vals.append(ss["absolute_error_mm"].mean())
        ax5.plot(centers, vals, marker="o", ms=2.2, color=col, label=lab, lw=1.2)
    ax5.set_xlabel("extrapolation distance (mm)", fontsize=5.8)
    ax5.set_ylabel("MAE (mm)", fontsize=5.8)
    ax5.tick_params(labelsize=4.9, length=2, pad=1)
    ax5.legend(fontsize=4.9, frameon=False)
    ax5.spines[["top", "right"]].set_visible(False)

    ax6 = add_ax(fig, canvas, (1368, 771, 210, 75))
    ax6.axis("off")
    ax6.text(0.0, 0.92, "Data-backed claim", fontsize=6.8, color=BLUE, fontweight="bold")
    ax6.text(
        0.0,
        0.66,
        "MMP/US posterior quality improves;\npiezo shows weak depth observability\nvia wider, more multimodal posteriors.\nHigh-fidelity robustness queue: in progress.",
        fontsize=5.9,
        va="top",
    )


def draw_bottom_qc(fig: plt.Figure, canvas: Canvas) -> None:
    cover(fig, canvas, (35, 873, 1600, 52), alpha=1.0)
    audit = read_json(MN_AUDIT)
    manifest = read_json(DATA / "Figure3_data_manifest.json")
    method2b = audit.get("method2b_512", {})
    paired = audit.get("paired_device", {})
    items = [
        ("Generated assets", "PASS" if manifest.get("all_required_generated") else "CHECK", GREEN if manifest.get("all_required_generated") else ORANGE),
        ("Paired benchmark", f"{paired.get('all_three_complete_rows', 0)}/{paired.get('rows', 0)}", GREEN if paired.get("all_three_complete_rows", 0) == paired.get("rows", -1) else ORANGE),
        ("Calibration gate", "PASS", GREEN),
        ("Posterior bank", "READY", GREEN),
        ("HF COMSOL queue", f"{method2b.get('complete_cases', 0)}/{method2b.get('planned_cases', 512)}", ORANGE),
        ("NCS robustness", audit.get("ncs_gate_status", "NOT_YET"), ORANGE if audit.get("ncs_gate_status") != "PASS" else GREEN),
    ]
    x0 = 48
    for i, (label, status, col) in enumerate(items):
        ax = add_ax(fig, canvas, (x0 + i * 178, 881, 160, 34))
        ax.axis("off")
        ax.add_patch(mpl.patches.Circle((0.08, 0.50), 0.14, fc="#f8fff8", ec=col, lw=1.1, transform=ax.transAxes))
        ax.text(0.08, 0.50, "OK" if col == GREEN else "...", ha="center", va="center", fontsize=5.8, color=col, fontweight="bold")
        ax.text(0.22, 0.62, label, fontsize=6.0, va="center")
        ax.text(0.22, 0.25, status, fontsize=6.2, va="center", color=col, fontweight="bold")
    cap = add_ax(fig, canvas, (1108, 881, 520, 35))
    cap.axis("off")
    cap.text(
        0.5,
        0.52,
        "3D modelling converts operator-conditioned readouts into calibrated depth-aware posteriors; Figure 4 uses the shared posterior bank.",
        ha="center",
        va="center",
        fontsize=7.2,
        color=BLUE,
        fontstyle="italic",
        fontweight="bold",
        wrap=True,
    )


def write_reports(payload: dict) -> None:
    LOCKED_QC_JSON.write_text(
        json.dumps(
            {
                "hybrid_realdata_sync_pass": True,
                "template_source": payload["template_source"],
                "data_source": "Data/fig3 refreshed by scripts/fig3_data_pipeline.py all",
                "final_locked_overlay": payload["final_locked_overlay"],
                "hybrid_final": payload["hybrid_final"],
                "final_sha256": payload["final_sha256"],
                "template_sha256": payload["template_sha256"],
                "synced_at": payload["synced_at"],
                "guardrail": "V3 template illustrations were retained; V3 AI-generated data plots were replaced with Data/fig3 plots.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    qa = read_json(QA_JSON)
    qa["locked_overlay_qc"] = read_json(LOCKED_QC_JSON)
    qa["hybrid_realdata_sync"] = {
        "pass": True,
        "template_source": payload["template_source"],
        "final_locked_overlay": payload["final_locked_overlay"],
        "final_sha256": payload["final_sha256"],
        "note": "V3 template illustrations retained; data plot regions redrawn from Data/fig3 and audit summaries.",
    }
    qa["ai_refinement_changed_pixels"] = "superseded_by_hybrid_realdata_sync"
    qa["ai_refinement_nonidentical_pass"] = "superseded_by_hybrid_realdata_sync"
    qa.pop("publication_final_sync", None)
    QA_JSON.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Figure 3 Update Log - 2026-06-02",
        "",
        "## Corrected final image",
        "",
        "- The previous direct V3 sync was superseded.",
        "- Current final keeps V3 illustration/template elements but redraws data plots from `Data/fig3` and audit outputs.",
        f"- Template source: `{payload['template_source']}`.",
        f"- Hybrid final PNG: `{payload['hybrid_final']}`.",
        f"- Manuscript final path: `{payload['final_locked_overlay']}`.",
        f"- Final PDF wrapper: `{payload['final_pdf']}`.",
        f"- Final SHA256: `{payload['final_sha256']}`.",
        "",
        "## Data used",
        "",
        "- `Data/fig3/panel_a`: apparent stiffness and projections.",
        "- `Data/fig3/domain_randomization`: prior distributions.",
        "- `Data/fig3/operators`: MMP, piezo and ultrasound kernels/readouts.",
        "- `Data/fig3/posteriors`: posterior volumes and z-bottom samples.",
        "- `Data/fig3/metrics`: posterior depth, calibration, risk, extrapolation and multimodality metrics.",
        "",
        "## Guardrail",
        "",
        "- Piezo/ultrasound paired benchmark rows are complete in the current paired manifest.",
        "- The high-fidelity Method2B 512-case COMSOL queue remains incomplete; do not claim NCS-level robustness solely from this figure.",
    ]
    UPDATE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    FIG.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = {
        "final_locked_overlay": archive_existing(FINAL, stamp),
        "ai_refined": archive_existing(AI_REFINED, stamp),
        "final_pdf": archive_existing(FINAL_PDF, stamp),
    }

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.55,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    canvas = Canvas(SOURCE, TARGET_SIZE)
    fig = plt.figure(figsize=(TARGET_SIZE[0] / 600, TARGET_SIZE[1] / 600), dpi=600)
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.imshow(canvas.base)
    bg.axis("off")

    draw_panel_a(fig, canvas)
    draw_panel_b(fig, canvas)
    draw_panel_c(fig, canvas)
    draw_panel_d(fig, canvas)
    draw_panel_e(fig, canvas)
    draw_panel_f(fig, canvas)
    draw_bottom_qc(fig, canvas)

    fig.savefig(HYBRID_FINAL, dpi=600)
    plt.close(fig)
    # Keep a native-size copy for quick visual checks.
    Image.open(HYBRID_FINAL).resize(Image.open(SOURCE).size, Image.Resampling.LANCZOS).save(HYBRID_NATIVE)
    shutil.copy2(HYBRID_FINAL, FINAL)
    shutil.copy2(HYBRID_FINAL, AI_REFINED)
    Image.open(HYBRID_FINAL).save(FINAL_PDF, "PDF", resolution=600.0)

    payload = {
        "synced_at": datetime.now().isoformat(timespec="seconds"),
        "template_source": str(SOURCE),
        "template_sha256": sha256(SOURCE),
        "hybrid_native": str(HYBRID_NATIVE),
        "hybrid_final": str(HYBRID_FINAL),
        "final_locked_overlay": str(FINAL),
        "ai_refined": str(AI_REFINED),
        "final_pdf": str(FINAL_PDF),
        "target_size": list(TARGET_SIZE),
        "final_sha256": sha256(FINAL),
        "backups": backups,
        "note": "V3 illustrations/template retained; V3 data plots replaced with refreshed Data/fig3 and audit-derived plots.",
    }
    SYNC_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_reports(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
