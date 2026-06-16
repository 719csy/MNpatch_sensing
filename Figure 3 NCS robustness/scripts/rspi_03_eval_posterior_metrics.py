from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from step3c_train_batch_benchmark import DiffusionDenoiser, SRCNN, SmallUNet, SurrogateForward, sample_diffusion as sample_direct_diffusion  # noqa: E402
from step3d_feature_utils import AUDIT_DIR, io_path  # noqa: E402
from step3j_method1_train_canonical_diffusion_cv import (  # noqa: E402
    CanonicalDataset,
    NormStats,
    aggregate_results,
    compute_norm_stats,
    constant_mean_metrics,
    evaluate_regressor,
    load_cache,
    make_fig3_true_ood_folds,
    make_group_folds,
    metric_summary,
    pick_bz_srcnn_channel,
    sample_diffusion_img2img,
    sample_residual_diffusion,
    set_seed,
)


EPS = 1e-8


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
        handle.write("\n")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint payload: {path}")


def load_norm_stats(fold_dir: Path, cache: dict[str, np.ndarray], train_idx: list[int]) -> NormStats:
    stats_path = fold_dir / "normalization_stats.json"
    if not stats_path.exists():
        return compute_norm_stats(cache, train_idx)
    payload = read_json(stats_path)
    return NormStats(
        input_mean=np.asarray(payload["input_mean"], dtype=np.float32),
        input_std=np.maximum(np.asarray(payload["input_std"], dtype=np.float32), 1e-6),
        target_low=float(payload["target_low_mu_logE"]),
        target_high=float(payload["target_high_mu_logE"]),
        target_mean_norm=float(payload["target_mean_norm"]),
    )


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy()
    yr = pd.Series(y[mask]).rank(method="average").to_numpy()
    sx = xr.std()
    sy = yr.std()
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    sum_pos = float(ranks[labels].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def crps_ensemble(samples: torch.Tensor, target: torch.Tensor) -> float:
    # samples: K,N,1,H,W. Uses sorted formula for E|Xi-Xj| without materializing KxK pairs.
    samples = samples.float()
    target = target.float().unsqueeze(0)
    term1 = torch.mean(torch.abs(samples - target))
    sorted_samples, _ = torch.sort(samples, dim=0)
    k = sorted_samples.shape[0]
    idx = torch.arange(1, k + 1, device=samples.device, dtype=samples.dtype).view(k, 1, 1, 1, 1)
    coeff = 2.0 * idx - float(k) - 1.0
    pairwise_mean_abs = (2.0 / float(k * k)) * torch.sum(coeff * sorted_samples, dim=0)
    term2 = 0.5 * torch.mean(pairwise_mean_abs)
    return float((term1 - term2).detach().cpu().item())


def uncertainty_metrics(samples: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    samples = samples.float().cpu()
    target = target.float().cpu()
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=False)
    abs_err = torch.abs(mean - target)
    sq_err = (mean - target) ** 2
    out: dict[str, float] = {}
    out["crps_norm"] = crps_ensemble(samples, target)
    for lo, hi, name in [(0.25, 0.75, "50"), (0.10, 0.90, "80"), (0.05, 0.95, "90"), (0.025, 0.975, "95")]:
        q_lo = torch.quantile(samples, lo, dim=0)
        q_hi = torch.quantile(samples, hi, dim=0)
        out[f"cov{name}"] = float(((target >= q_lo) & (target <= q_hi)).float().mean().item())
        out[f"width{name}_norm"] = float((q_hi - q_lo).mean().item())
    flat_std = std.numpy().ravel()
    flat_abs = abs_err.numpy().ravel()
    flat_sq = sq_err.numpy().ravel()
    out["uncertainty_error_spearman"] = rank_corr(flat_std, flat_abs)
    threshold = np.quantile(flat_abs[np.isfinite(flat_abs)], 0.90) if np.isfinite(flat_abs).any() else np.nan
    out["failure_auroc_top10_abs_error"] = auc_binary(flat_abs >= threshold, flat_std)
    out.update(calibration_errors(flat_std, flat_sq, n_bins=10))
    out["ause_abs_error"] = ause(flat_abs, flat_std)
    out["pit_mean"] = float((samples <= target.unsqueeze(0)).float().mean().item())
    out["posterior_std_norm_mean"] = float(std.mean().item())
    return out


def calibration_errors(std: np.ndarray, sq_err: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    std = np.asarray(std, dtype=np.float64)
    sq_err = np.asarray(sq_err, dtype=np.float64)
    mask = np.isfinite(std) & np.isfinite(sq_err)
    std = std[mask]
    sq_err = sq_err[mask]
    if len(std) < n_bins:
        return {"uce": float("nan"), "ence": float("nan")}
    if float(np.nanmean(std)) <= EPS:
        return {"uce": float("nan"), "ence": float("nan")}
    order = np.argsort(std)
    bins = np.array_split(order, n_bins)
    uce = 0.0
    ence = 0.0
    total = float(len(std))
    for b in bins:
        if len(b) == 0:
            continue
        conf = float(np.mean(std[b]))
        rmse = float(math.sqrt(max(np.mean(sq_err[b]), 0.0)))
        w = len(b) / total
        uce += w * abs(rmse - conf)
        ence += w * abs(rmse - conf) / max(conf, EPS)
    return {"uce": float(uce), "ence": float(ence)}


def ause(abs_err: np.ndarray, uncertainty: np.ndarray, n_steps: int = 50) -> float:
    abs_err = np.asarray(abs_err, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    mask = np.isfinite(abs_err) & np.isfinite(uncertainty)
    abs_err = abs_err[mask]
    uncertainty = uncertainty[mask]
    if len(abs_err) < 10:
        return float("nan")
    model_order = np.argsort(-uncertainty)
    oracle_order = np.argsort(-abs_err)
    fracs = np.linspace(0.0, 0.95, n_steps)
    diffs = []
    for frac in fracs:
        keep = max(int(round((1.0 - frac) * len(abs_err))), 1)
        model_err = float(np.mean(abs_err[model_order[-keep:]]))
        oracle_err = float(np.mean(abs_err[oracle_order[-keep:]]))
        diffs.append(max(model_err - oracle_err, 0.0))
    return float(np.trapezoid(diffs, fracs) / 0.95)


def deterministic_as_posterior_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    samples = pred.unsqueeze(0).repeat(2, 1, 1, 1, 1)
    return uncertainty_metrics(samples, target)


def collect_regressor_predictions(model_name: str, model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    logs: list[torch.Tensor] = []
    kpas: list[torch.Tensor] = []
    condition_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["srcnn_input"] if model_name == "srcnn" else batch["input"]
            pred = model(x.to(device))
            if model_name != "srcnn":
                pred = torch.sigmoid(pred)
            preds.append(pred.detach().cpu())
            targets.append(batch["target"].cpu())
            logs.append(batch["target_log_raw"].cpu())
            kpas.append(batch["target_kpa_raw"].cpu())
            condition_ids.extend(batch["condition_id"])
    return torch.cat(preds), torch.cat(targets), torch.cat(logs), torch.cat(kpas), condition_ids


def eval_samples(samples_by_batch: list[torch.Tensor], targets_by_batch: list[torch.Tensor], logs_by_batch: list[torch.Tensor], kpas_by_batch: list[torch.Tensor], condition_ids: list[str]) -> dict[str, float]:
    samples = torch.cat(samples_by_batch, dim=1)
    targets = torch.cat(targets_by_batch)
    logs = torch.cat(logs_by_batch)
    kpas = torch.cat(kpas_by_batch)
    mean = samples.mean(dim=0)
    out = metric_summary(mean, targets, logs, kpas, condition_ids)
    out.update(uncertainty_metrics(samples, targets))
    return out


def collect_sample_batches(loader: DataLoader, sample_fn) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[str]]:
    samples_by_batch: list[torch.Tensor] = []
    targets_by_batch: list[torch.Tensor] = []
    logs_by_batch: list[torch.Tensor] = []
    kpas_by_batch: list[torch.Tensor] = []
    condition_ids: list[str] = []
    for batch in loader:
        samples = sample_fn(batch)
        samples_by_batch.append(samples)
        targets_by_batch.append(batch["target"].cpu())
        logs_by_batch.append(batch["target_log_raw"].cpu())
        kpas_by_batch.append(batch["target_kpa_raw"].cpu())
        condition_ids.extend(batch["condition_id"])
    return samples_by_batch, targets_by_batch, logs_by_batch, kpas_by_batch, condition_ids


def conformal_interval_scale(
    samples_by_batch: list[torch.Tensor],
    targets_by_batch: list[torch.Tensor],
    target_coverage: float = 0.90,
    max_scale: float = 50.0,
) -> float:
    ratios: list[np.ndarray] = []
    for samples, target in zip(samples_by_batch, targets_by_batch, strict=False):
        samples = samples.float().cpu()
        target = target.float().cpu()
        mean = samples.mean(dim=0)
        q_lo = torch.quantile(samples, 0.05, dim=0)
        q_hi = torch.quantile(samples, 0.95, dim=0)
        denom_lo = torch.clamp(mean - q_lo, min=EPS)
        denom_hi = torch.clamp(q_hi - mean, min=EPS)
        below = target < mean
        needed = torch.where(below, (mean - target) / denom_lo, (target - mean) / denom_hi)
        needed = torch.clamp(needed, min=0.0)
        arr = needed.numpy().ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            ratios.append(arr)
    if not ratios:
        return 1.0
    all_ratios = np.concatenate(ratios)
    q = float(np.quantile(all_ratios, float(target_coverage)))
    return float(np.clip(max(1.0, q), 1.0, float(max_scale)))


def scale_sample_batches(samples_by_batch: list[torch.Tensor], scale: float) -> list[torch.Tensor]:
    scaled: list[torch.Tensor] = []
    for samples in samples_by_batch:
        mean = samples.mean(dim=0, keepdim=True)
        scaled.append(mean + float(scale) * (samples - mean))
    return scaled


def append_calibrated_row(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    model_name: str,
    val_batches: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[str]] | None,
    test_batches: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[str]],
    split_payload: dict[str, Any],
    residual_scale: float | None = None,
) -> None:
    if args.coverage_calibration != "validation" or val_batches is None:
        return
    val_samples, val_targets, *_ = val_batches
    test_samples, test_targets, test_logs, test_kpas, test_ids = test_batches
    scale = conformal_interval_scale(
        val_samples,
        val_targets,
        target_coverage=float(args.calibration_coverage),
        max_scale=float(args.calibration_max_scale),
    )
    row = {
        "model": f"{model_name}_calibrated90",
        **eval_samples(scale_sample_batches(test_samples, scale), test_targets, test_logs, test_kpas, test_ids),
        **split_payload,
    }
    row["calibration_split"] = "validation"
    row["calibration_target_coverage"] = float(args.calibration_coverage)
    row["calibration_scale_90"] = scale
    if residual_scale is not None:
        row["residual_scale"] = float(residual_scale)
    rows.append(row)


def evaluate_fold(
    args: argparse.Namespace,
    split_mode: str,
    fold: dict[str, Any],
    manifest: pd.DataFrame,
    cache: dict[str, np.ndarray],
    channel_names: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    fold_dir = args.training_dir / split_mode / f"fold_{fold['fold']:02d}"
    if not fold_dir.exists():
        raise RuntimeError(f"Missing fold dir: {fold_dir}")
    stats = load_norm_stats(fold_dir, cache, fold["train_idx"])
    srcnn_channel = pick_bz_srcnn_channel(channel_names)
    test_ds = CanonicalDataset(manifest, cache, fold["test_idx"], stats, srcnn_channel, augment=False)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    val_loader: DataLoader | None = None
    if args.coverage_calibration == "validation":
        val_ds = CanonicalDataset(manifest, cache, fold["val_idx"], stats, srcnn_channel, augment=False)
        val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    split_payload = {
        "dataset": args.dataset_name,
        "split_mode": split_mode,
        "fold": int(fold["fold"]),
        "split_column": fold["split_column"],
        "test_group": fold["test_group"],
        "train_count": int(len(fold["train_idx"])),
        "val_count": int(len(fold["val_idx"])),
        "test_count": int(len(fold["test_idx"])),
    }
    only_models = {x.strip() for x in str(args.only_models).split(",") if x.strip()}

    def want_model(name: str) -> bool:
        return not only_models or name in only_models

    rows: list[dict[str, Any]] = []
    if want_model("train_mean_prior"):
        rows.append({"model": "train_mean_prior", **constant_mean_metrics(test_loader, stats.target_mean_norm), **split_payload})

    if want_model("srcnn_bz_single_channel"):
        srcnn = SRCNN()
        srcnn.load_state_dict(load_state(fold_dir / "srcnn_best.pt"))
        srcnn.to(device)
        srcnn_pred, target, logs, kpas, ids = collect_regressor_predictions("srcnn", srcnn, test_loader, device)
        row = {"model": "srcnn_bz_single_channel", **metric_summary(srcnn_pred, target, logs, kpas, ids), **deterministic_as_posterior_metrics(srcnn_pred, target), **split_payload}
        rows.append(row)

    unet = SmallUNet(cache["input64"].shape[1], out_ch=1, base=int(args.base_channels))
    unet.load_state_dict(load_state(fold_dir / "unet_best.pt"))
    unet.to(device)
    if want_model("unet_canonical_bxyz"):
        unet_pred, target, logs, kpas, ids = collect_regressor_predictions("unet", unet, test_loader, device)
        row = {"model": "unet_canonical_bxyz", **metric_summary(unet_pred, target, logs, kpas, ids), **deterministic_as_posterior_metrics(unet_pred, target), **split_payload}
        rows.append(row)

    surrogate = SurrogateForward()
    surrogate.load_state_dict(load_state(fold_dir / "surrogate_forward.pt"))
    surrogate.to(device)
    surrogate.eval()

    diffusion = DiffusionDenoiser(cond_ch=cache["input64"].shape[1], timesteps=int(args.timesteps), base=int(args.base_channels))
    diffusion.load_state_dict(load_state(fold_dir / "vanilla_diffusion.pt"))
    diffusion.to(device)
    for model_name, guidance_surrogate, guidance_scale in [
        ("vanilla_diffusion", None, 0.0),
        ("likelihood_guided_diffusion", surrogate, float(args.guidance_scale)),
    ]:
        if not (want_model(model_name) or want_model(f"{model_name}_calibrated90")):
            continue

        def sample_fn(batch: dict[str, Any]) -> torch.Tensor:
            return sample_direct_diffusion(
                diffusion,
                batch,
                device,
                int(args.timesteps),
                False,
                int(args.posterior_samples),
                surrogate=guidance_surrogate,
                guidance_scale=guidance_scale,
                optical_prior_scale=0.0,
            )

        test_batches = collect_sample_batches(test_loader, sample_fn)
        if want_model(model_name):
            rows.append({"model": model_name, **eval_samples(*test_batches), **split_payload})
        val_batches = collect_sample_batches(val_loader, sample_fn) if val_loader is not None else None
        if want_model(f"{model_name}_calibrated90"):
            append_calibrated_row(rows, args, model_name, val_batches, test_batches, split_payload)

    for model_name, guidance_surrogate, guidance_scale in [
        ("prior_init_diffusion_unet", None, 0.0),
        ("residual_likelihood_guided_diffusion", surrogate, float(args.guidance_scale)),
    ]:
        if not (want_model(model_name) or want_model(f"{model_name}_calibrated90")):
            continue

        def sample_fn(batch: dict[str, Any]) -> torch.Tensor:
            return sample_diffusion_img2img(
                diffusion,
                unet,
                batch,
                device,
                int(args.timesteps),
                int(args.img2img_start_step),
                False,
                int(args.posterior_samples),
                surrogate=guidance_surrogate,
                guidance_scale=guidance_scale,
            )

        test_batches = collect_sample_batches(test_loader, sample_fn)
        if want_model(model_name):
            rows.append({"model": model_name, **eval_samples(*test_batches), **split_payload})
        val_batches = collect_sample_batches(val_loader, sample_fn) if val_loader is not None else None
        if want_model(f"{model_name}_calibrated90"):
            append_calibrated_row(rows, args, model_name, val_batches, test_batches, split_payload)

    residual_checkpoint = torch.load(fold_dir / "explicit_residual_target_diffusion.pt", map_location="cpu", weights_only=False)
    residual_scale = float(residual_checkpoint.get("residual_scale", args.residual_scale))
    residual_diffusion = DiffusionDenoiser(cond_ch=cache["input64"].shape[1], timesteps=int(args.timesteps), base=int(args.base_channels))
    residual_diffusion.load_state_dict(residual_checkpoint["model"])
    residual_diffusion.to(device)
    for model_name, guidance_surrogate, guidance_scale in [
        ("explicit_residual_target_diffusion", None, 0.0),
        ("explicit_residual_likelihood_guided_diffusion", surrogate, float(args.guidance_scale)),
    ]:
        if not (want_model(model_name) or want_model(f"{model_name}_calibrated90")):
            continue

        def sample_fn(batch: dict[str, Any]) -> torch.Tensor:
            return sample_residual_diffusion(
                residual_diffusion,
                unet,
                batch,
                device,
                int(args.timesteps),
                residual_scale,
                int(args.posterior_samples),
                surrogate=guidance_surrogate,
                guidance_scale=guidance_scale,
            )

        test_batches = collect_sample_batches(test_loader, sample_fn)
        row = {"model": model_name, **eval_samples(*test_batches), **split_payload}
        row["residual_scale"] = residual_scale
        if want_model(model_name):
            rows.append(row)
        val_batches = collect_sample_batches(val_loader, sample_fn) if val_loader is not None else None
        if want_model(f"{model_name}_calibrated90"):
            append_calibrated_row(rows, args, model_name, val_batches, test_batches, split_payload, residual_scale=residual_scale)
    return rows


def aggregate_posterior(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    if "ence" in metrics.columns:
        degenerate = pd.Series(False, index=metrics.index)
        for col in ("width90_norm", "posterior_std_norm_mean"):
            if col in metrics.columns:
                values = pd.to_numeric(metrics[col], errors="coerce")
                degenerate = degenerate | (values.notna() & (values <= 1e-12))
        metrics.loc[degenerate, "ence"] = np.nan
    rows: list[dict[str, Any]] = []
    numeric_cols = [
        c
        for c in metrics.columns
        if c
        not in {
            "dataset",
            "split_mode",
            "model",
            "fold",
            "split_column",
            "test_group",
            "condition_ids",
        }
        and pd.api.types.is_numeric_dtype(metrics[c])
    ]
    for keys, sub in metrics.groupby(["dataset", "split_mode", "model"], sort=True):
        dataset, split_mode, model = keys
        row: dict[str, Any] = {"dataset": dataset, "split_mode": split_mode, "model": model, "n_folds": int(len(sub))}
        for col in numeric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "split_mode", "rmse_norm_mean", "model"], na_position="last")


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    manifest = pd.read_csv(io_path(args.manifest)).reset_index(drop=True)
    cache, channel_names = load_cache(args.cache)
    if len(manifest) != cache["input64"].shape[0]:
        raise RuntimeError("Manifest row count does not match cache sample count.")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_modes = [x.strip() for x in str(args.split_modes).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for split_mode in split_modes:
        if split_mode == "leave_prior_group":
            folds = make_group_folds(manifest, "split_group_id")
        elif split_mode == "leave_disease_family":
            folds = make_group_folds(manifest, "disease")
        elif split_mode.startswith("leave_column__"):
            split_column = split_mode.split("__", 1)[1]
            if split_column not in manifest.columns:
                raise ValueError(f"Requested split column not found in manifest: {split_column}")
            folds = make_group_folds(manifest, split_column)
        elif split_mode == "fig3_true_ood":
            folds = make_fig3_true_ood_folds(manifest)
        else:
            raise ValueError(f"Unknown split mode: {split_mode}")
        if int(args.max_folds) > 0:
            folds = folds[: int(args.max_folds)]
        for fold in folds:
            print(f"FOLD_START split_mode={split_mode} fold={fold}", flush=True)
            fold_rows = evaluate_fold(args, split_mode, fold, manifest, cache, channel_names, device)
            rows.extend(fold_rows)
            pd.DataFrame(rows).to_csv(io_path(out_dir / "rspi_posterior_metrics.csv"), index=False)
            print(f"FOLD_DONE split_mode={split_mode} fold={fold} total_rows={len(rows)}", flush=True)
    metrics = pd.DataFrame(rows)
    aggregate = aggregate_posterior(metrics)
    metrics_path = out_dir / "rspi_posterior_metrics.csv"
    aggregate_path = out_dir / "rspi_posterior_aggregate.csv"
    metrics.to_csv(io_path(metrics_path), index=False)
    aggregate.to_csv(io_path(aggregate_path), index=False)
    best = aggregate.sort_values(["dataset", "split_mode", "rmse_norm_mean"]).groupby(["dataset", "split_mode"]).head(1).to_dict("records")
    summary = {
        "step": "rspi_03_eval_posterior_metrics",
        "decision": "RSPI_POSTERIOR_METRICS_COMPLETED",
        "dataset": args.dataset_name,
        "device": str(device),
        "sample_count": int(len(manifest)),
        "input_shape": list(cache["input64"].shape),
        "target_shape": list(cache["target_logE64"].shape),
        "training_dir": str(args.training_dir),
        "posterior_samples": int(args.posterior_samples),
        "timesteps": int(args.timesteps),
        "img2img_start_step": int(args.img2img_start_step),
        "base_channels": int(args.base_channels),
        "guidance_scale": float(args.guidance_scale),
        "coverage_calibration": args.coverage_calibration,
        "calibration_coverage": float(args.calibration_coverage),
        "split_modes": split_modes,
        "fold_rows": int(len(metrics)),
        "best_by_split_rmse": best,
        "outputs": {
            "metrics_csv": str(metrics_path),
            "aggregate_csv": str(aggregate_path),
            "summary_json": str(out_dir / "rspi_posterior_metrics_summary.json"),
        },
    }
    write_json(out_dir / "rspi_posterior_metrics_summary.json", summary)
    lines = [
        "# RSPI Posterior Metrics",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Dataset: `{args.dataset_name}`",
        f"- Posterior samples: `{args.posterior_samples}`",
        f"- Metrics: `CRPS, Cov50/80/90/95, Width90, UCE, ENCE, AUSE, AUROC_fail, uncertainty-error Spearman`",
        f"- Aggregate: `{aggregate_path}`",
    ]
    (out_dir / "rspi_posterior_metrics_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RSPI posterior metrics from existing CV checkpoints.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-modes", default="leave_prior_group,leave_disease_family")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--posterior-samples", type=int, default=64)
    parser.add_argument("--img2img-start-step", type=int, default=10)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--guidance-scale", type=float, default=0.06)
    parser.add_argument("--coverage-calibration", choices=["none", "validation"], default="none")
    parser.add_argument("--calibration-coverage", type=float, default=0.90)
    parser.add_argument("--calibration-max-scale", type=float, default=50.0)
    parser.add_argument("--only-models", default="", help="Optional comma-separated model allow-list, including calibrated90 names.")
    parser.add_argument("--seed", type=int, default=23)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
