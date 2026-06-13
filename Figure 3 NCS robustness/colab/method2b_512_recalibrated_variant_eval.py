from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath

import pandas as pd

from figure3_colab_orchestrator import (
    DRIVE_MODE,
    MANUSCRIPT_DRIVE_REL,
    MN_DRIVE_REL,
    MN_WORK,
    WORK_ROOT,
    ApiDriveAccess,
    MountedDriveAccess,
    rel_to_log_path,
    rel_to_windows_path,
    run_cmd,
)


RUN_TAG = os.environ.get("METHOD2B_RECAL_RUN_TAG") or "20260613_recalibrated_9model"
SEEDS = [int(x.strip()) for x in os.environ.get("METHOD2B_RECAL_SEEDS", "1,2,3,4,5").split(",") if x.strip()]
SPLIT_MODES = os.environ.get("METHOD2B_RECAL_SPLIT_MODES", "fig3_true_ood")
BATCH_SIZE = os.environ.get("METHOD2B_RECAL_BATCH_SIZE", "16")
BASE_CHANNELS = os.environ.get("METHOD2B_RECAL_BASE_CHANNELS", "16")
TIMESTEPS = os.environ.get("METHOD2B_RECAL_TIMESTEPS", "200")
POSTERIOR_SAMPLES = os.environ.get("METHOD2B_RECAL_POSTERIOR_SAMPLES", "64")
IMG2IMG_START_STEP = os.environ.get("METHOD2B_RECAL_IMG2IMG_START_STEP", "10")
GUIDANCE_SCALE = os.environ.get("METHOD2B_RECAL_GUIDANCE_SCALE", "0.06")
RESIDUAL_SCALE = os.environ.get("METHOD2B_RECAL_RESIDUAL_SCALE", "0.25")
CALIBRATION_COVERAGE = os.environ.get("METHOD2B_RECAL_COVERAGE", "0.90")
CALIBRATION_MAX_SCALE = os.environ.get("METHOD2B_RECAL_MAX_SCALE", "50")
EVAL_DIR_NAME = os.environ.get("METHOD2B_RECAL_EVAL_DIR", f"posterior_recalibrated_9model_{RUN_TAG}")
AGG_DIR_NAME = os.environ.get("METHOD2B_RECAL_AGG_DIR", f"aggregate_recalibrated_9model_{RUN_TAG}")
LOGS_DIR_NAME = os.environ.get("METHOD2B_RECAL_LOGS_DIR", f"logs_recalibrated_9model_{RUN_TAG}")

SOURCE_RUN_DRIVE_REL = PurePosixPath(
    os.environ.get(
        "METHOD2B_RECAL_SOURCE_RUN_DRIVE_REL",
        "MN_simulation/outputs/rspi_joint_052026/method2b_512_five_seed_ood_full512_20260611",
    )
)
OUTPUT_DRIVE_REL = PurePosixPath(os.environ.get("METHOD2B_RECAL_OUTPUT_DRIVE_REL", SOURCE_RUN_DRIVE_REL.as_posix()))
CACHE_DRIVE_REL = PurePosixPath(
    os.environ.get(
        "METHOD2B_RECAL_CACHE_DRIVE_REL",
        "MN_simulation/outputs/rspi_joint_052026/method2b_512_targetmap_diffusion_full512_20260611",
    )
)
CACHE_NAMES = [
    "method2b_512_diffusion_ready_cache.npz",
    "method2b_512_diffusion_ready_manifest.csv",
    "method2b_512_diffusion_cache_summary.json",
    "method2b_512_diffusion_excluded_cases.csv",
]
SCRIPT_NAMES = [
    "rspi_03_eval_posterior_metrics.py",
    "step3j_method1_train_canonical_diffusion_cv.py",
    "step3c_train_batch_benchmark.py",
    "step3d_feature_utils.py",
]
PATCHED_RSPI_URL = os.environ.get("METHOD2B_RECAL_PATCHED_RSPI_URL", "").strip()


def build_drive_access() -> ApiDriveAccess | MountedDriveAccess:
    if DRIVE_MODE == "mount":
        print("Using mounted Google Drive filesystem access.", flush=True)
        return MountedDriveAccess()
    if DRIVE_MODE != "api":
        raise ValueError("FIG3_COLAB_DRIVE_MODE must be 'api' or 'mount'.")
    print("Using Google Drive API access.", flush=True)
    return ApiDriveAccess()


def logged_cmd(cmd: list[object], log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\nRUN:", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("COMMAND\n")
        handle.write(" ".join(shlex.quote(str(x)) for x in cmd) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(map(str, cmd))}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def check_cuda() -> str:
    run_cmd(["nvidia-smi"])
    import torch

    print("torch", torch.__version__, flush=True)
    print("torch.version.cuda", torch.version.cuda, flush=True)
    print("torch.cuda.is_available", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use Runtime > Change runtime type > GPU, then rerun.")
    gpu = torch.cuda.get_device_name(0)
    print("GPU", gpu, flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return gpu


def download_required_scripts(drive_access: ApiDriveAccess | MountedDriveAccess) -> None:
    scripts_dir = MN_WORK / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("METHOD2B_RECAL_DOWNLOAD_ALL_SCRIPTS", "0").strip().lower() in {"1", "true", "yes"}:
        drive_access.download_folder(MN_DRIVE_REL / "scripts", scripts_dir)
        return
    for name in SCRIPT_NAMES:
        drive_access.download_file(MN_DRIVE_REL / "scripts" / name, scripts_dir / name)
    if PATCHED_RSPI_URL:
        dst = scripts_dir / "rspi_03_eval_posterior_metrics.py"
        print(f"Downloading patched posterior evaluator: {PATCHED_RSPI_URL}", flush=True)
        urllib.request.urlretrieve(PATCHED_RSPI_URL, dst)
        print(f"Patched posterior evaluator written to {dst}", flush=True)


def aggregate_seed_metrics(seed_eval_dirs: list[Path], out_dir: Path) -> tuple[Path, Path]:
    if str(MN_WORK / "scripts") not in sys.path:
        sys.path.insert(0, str(MN_WORK / "scripts"))
    from rspi_03_eval_posterior_metrics import aggregate_posterior

    frames: list[pd.DataFrame] = []
    for seed, eval_dir in zip(SEEDS, seed_eval_dirs, strict=True):
        metrics_path = eval_dir / "rspi_posterior_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        frame = pd.read_csv(metrics_path)
        if "seed" not in frame.columns:
            frame.insert(0, "seed", int(seed))
        frame.insert(0, "source_metrics_csv", str(metrics_path))
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    aggregate = aggregate_posterior(metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "method2b_512_five_seed_posterior_metrics.csv"
    aggregate_path = out_dir / "method2b_512_five_seed_posterior_aggregate.csv"
    summary_path = out_dir / "method2b_512_five_seed_posterior_aggregate_summary.json"
    metrics.to_csv(metrics_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    best = (
        aggregate.sort_values(["dataset", "split_mode", "rmse_norm_mean"])
        .groupby(["dataset", "split_mode"])
        .head(1)
        .to_dict("records")
    )
    write_json(
        summary_path,
        {
            "step": "method2b_512_recalibrated_variant_eval",
            "decision": "METHOD2B_512_RECALIBRATED_9MODEL_POSTERIOR_AGGREGATED",
            "run_tag": RUN_TAG,
            "seeds": SEEDS,
            "row_count": int(len(metrics)),
            "models": sorted(str(x) for x in metrics["model"].dropna().unique()) if "model" in metrics else [],
            "split_modes": sorted(str(x) for x in metrics["split_mode"].dropna().unique()) if "split_mode" in metrics else [],
            "posterior_samples": int(POSTERIOR_SAMPLES),
            "guidance_scale": float(GUIDANCE_SCALE),
            "coverage_calibration": "validation",
            "calibration_coverage": float(CALIBRATION_COVERAGE),
            "best_by_split": best,
            "outputs": {
                "metrics_csv": str(metrics_path),
                "aggregate_csv": str(aggregate_path),
                "summary_json": str(summary_path),
            },
        },
    )
    return metrics_path, aggregate_path


def main() -> None:
    print("METHOD2B_RECAL_RUN_TAG", RUN_TAG, flush=True)
    print("METHOD2B_RECAL_SEEDS", SEEDS, flush=True)
    print("FIG3_COLAB_DRIVE_MODE", DRIVE_MODE, flush=True)
    drive_access = build_drive_access()
    gpu = check_cuda()
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "tqdm"])

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    download_required_scripts(drive_access)

    cache_work_dir = MN_WORK / "outputs" / "rspi_joint_052026" / "method2b_512_targetmap_diffusion_full512_20260611"
    for name in CACHE_NAMES:
        drive_access.download_file(CACHE_DRIVE_REL / name, cache_work_dir / name)
    cache_path = cache_work_dir / "method2b_512_diffusion_ready_cache.npz"
    manifest_path = cache_work_dir / "method2b_512_diffusion_ready_manifest.csv"

    eval_script = MN_WORK / "scripts" / "rspi_03_eval_posterior_metrics.py"
    run_root = MN_WORK / "outputs" / "rspi_joint_052026" / f"method2b_512_recalibrated_9model_{RUN_TAG}"
    logs_dir = run_root / LOGS_DIR_NAME
    seed_eval_dirs: list[Path] = []
    seed_records: list[dict] = []
    for seed in SEEDS:
        seed_root = run_root / f"seed_{seed}"
        train_dir = seed_root / "training"
        eval_dir = seed_root / EVAL_DIR_NAME
        drive_access.download_folder(SOURCE_RUN_DRIVE_REL / f"seed_{seed}" / "training", train_dir)
        logged_cmd(
            [
                sys.executable,
                eval_script,
                "--dataset-name",
                "method2b_512_full_comsol_recalibrated_9model",
                "--cache",
                cache_path,
                "--manifest",
                manifest_path,
                "--training-dir",
                train_dir,
                "--out-dir",
                eval_dir,
                "--split-modes",
                SPLIT_MODES,
                "--device",
                "cuda",
                "--batch-size",
                BATCH_SIZE,
                "--base-channels",
                BASE_CHANNELS,
                "--timesteps",
                TIMESTEPS,
                "--posterior-samples",
                POSTERIOR_SAMPLES,
                "--img2img-start-step",
                IMG2IMG_START_STEP,
                "--residual-scale",
                RESIDUAL_SCALE,
                "--guidance-scale",
                GUIDANCE_SCALE,
                "--coverage-calibration",
                "validation",
                "--calibration-coverage",
                CALIBRATION_COVERAGE,
                "--calibration-max-scale",
                CALIBRATION_MAX_SCALE,
                "--seed",
                str(seed),
            ],
            logs_dir / f"seed_{seed}_posterior_recalibrated_9model.log",
            cwd=MN_WORK,
        )
        seed_eval_dirs.append(eval_dir)
        seed_records.append({"seed": seed, "training": str(train_dir), "posterior": str(eval_dir)})
        if os.environ.get("METHOD2B_RECAL_UPLOAD_AFTER_EACH_SEED", "1").strip().lower() not in {"0", "false", "no"}:
            drive_access.upload_folder(eval_dir, OUTPUT_DRIVE_REL / f"seed_{seed}" / EVAL_DIR_NAME)
            drive_access.upload_folder(logs_dir, OUTPUT_DRIVE_REL / LOGS_DIR_NAME)

    aggregate_dir = run_root / AGG_DIR_NAME
    metrics_path, aggregate_path = aggregate_seed_metrics(seed_eval_dirs, aggregate_dir)
    drive_access.upload_folder(aggregate_dir, OUTPUT_DRIVE_REL / AGG_DIR_NAME)
    drive_access.upload_folder(logs_dir, OUTPUT_DRIVE_REL / LOGS_DIR_NAME)

    pointer = {
        "run_tag": RUN_TAG,
        "target": "method2b_512_full_comsol_recalibrated_9model",
        "gpu": gpu,
        "drive_mode": DRIVE_MODE,
        "seeds": SEEDS,
        "posterior_samples": int(POSTERIOR_SAMPLES),
        "guidance_scale": float(GUIDANCE_SCALE),
        "coverage_calibration": "validation",
        "calibration_coverage": float(CALIBRATION_COVERAGE),
        "source_run_drive": rel_to_log_path(SOURCE_RUN_DRIVE_REL),
        "output_drive": rel_to_log_path(OUTPUT_DRIVE_REL),
        "aggregate_metrics": rel_to_log_path(OUTPUT_DRIVE_REL / AGG_DIR_NAME / metrics_path.name),
        "aggregate_csv": rel_to_log_path(OUTPUT_DRIVE_REL / AGG_DIR_NAME / aggregate_path.name),
        "aggregate_csv_windows": rel_to_windows_path(OUTPUT_DRIVE_REL / AGG_DIR_NAME / aggregate_path.name),
        "seed_runs": seed_records,
    }
    pointer_rel = MANUSCRIPT_DRIVE_REL / "reports" / f"method2b_512_recalibrated_9model_pointer_{RUN_TAG}.json"
    drive_access.upload_text(json.dumps(pointer, indent=2), pointer_rel)
    print("\nMETHOD2B 512 RECALIBRATED 9-MODEL EVAL COMPLETE", flush=True)
    print(json.dumps(pointer, indent=2), flush=True)


if __name__ == "__main__":
    main()
